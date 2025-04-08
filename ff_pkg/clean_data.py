import pandas as pd
import numpy as np

def clean_standings(standings_raw):
    clean_standings = pd.DataFrame(columns = ['season', 'reg_season_rank', 'final_rank', 'owner', 'wins', 'losses', 'win_pct', 'pts_for', 'pts_against'])
    for index, row in standings.iterrows():
        szn = row['season']
        szn_rank = int(row['regSeasonRank'])
        rank = int(row['rank'])
        owner = row['owner'].lower().capitalize()
        wins = int(row['teamRecord'].split('-')[0])
        loss = int(row['teamRecord'].split('-')[1])
        pct = float(row['teamWinPct'])
        pts_for = float(row['teamPts'].replace(',', ''))
        pts_against = float(row['teamPtsLast'].replace(',', ''))
    
        clean_standings.loc[len(clean_standings)] = [szn, szn_rank, rank, owner, wins, loss, pct, pts_for, pts_against]
    return clean_standings


def clean_weekly(weekly_raw):
    clean_weekly = pd.DataFrame(columns = ['season', 'owner', 'week', 'opponent', 'result', 'pts_for', 'pts_against'])
    
    for index, row in weekly.iterrows():
        szn = row['season']
        owner = row['owner'].lower().capitalize()
        week = int(row['week'])
        opp = row['opponent_owner'].lower().capitalize()
        res = row['result']
        if 'Win' in res:
            r = 1
        elif 'Loss' in res:
            r = 0
    
        pts_for = float(res.split(' - ')[0])
        pts_against = float(res.split(' - ')[1].split(' ')[0])
    
        clean_weekly.loc[len(clean_weekly)] = [szn, owner, week, opp, r, pts_for, pts_against]
    return clean_weekly


def full_history(weekly_raw):
    aggregate = pd.DataFrame(columns = ['owner', 'season', 'week', 'opponent', 'pts_for', 'pts_against', 
                              'result', 'szn_pts_for', 'szn_pts_against', 'szn_wins', 'szn_losses', 
                              'szn_pct', 'all_time_pts_for', 'all_time_pts_against', 'all_time_wins', 
                             'all_time_losses', 'all_time_pct', 'reg_szn_rank', 'avg_szn_rank', 'final_rank', 'avg_final_rank', 'championships'])
    weekly['owner'] = weekly['owner'].str.lower().str.capitalize()
    weekly['opponent_owner'] = weekly['opponent_owner'].str.lower().str.capitalize()
    
    for owner in weekly['owner'].unique():
        all_time_pts_for = 0
        all_time_pts_against = 0
        all_time_wins = 0
        all_time_losses = 0
        all_time_pct = 0
        avg_reg_rank = 0
        avg_final_rank = 0
        championships = 0
    
        szn_tracker = 1

        # season level measures
        for season in weekly['season'].unique():
            ppt_ref = weekly[(weekly['season'] == season) & (weekly['owner'] == owner)]
            szn_pts_for = 0
            szn_pts_against = 0
            szn_wins = 0
            szn_losses = 0
    
    
            szn_ref = clean_standings[(clean_standings['owner'] == owner) & (clean_standings['season'] == season)]
            rank = int(szn_ref['final_rank'].values[0])
            reg_rank = int(szn_ref['reg_season_rank'].values[0])
            
            if (rank == 1):
                championships += 1
                
            avg_final_rank = ((avg_final_rank * (szn_tracker - 1)) + rank) / szn_tracker
            avg_reg_rank = ((avg_reg_rank * (szn_tracker - 1)) + reg_rank) / szn_tracker
            
            szn_tracker += 1

            # week level measures
            for index, row in ppt_ref.iterrows():
                owner = row['owner']
                week = int(row['week'])
                opp = row['opponent_owner']
                res = row['result']
                pts_for = float(res.split(' - ')[0])
                pts_ag = float(res.split(' - ')[1].split(' ')[0])
                if 'Win' in res:
                    r = 1
                elif 'Loss' in res:
                    r = 0
    
                szn_pts_for += pts_for
                szn_pts_against += pts_ag
                all_time_pts_for += pts_for
                all_time_pts_against += pts_ag
    
                szn_wins += r
                szn_losses += (1 - r) 
                szn_pct = szn_wins/(szn_wins + szn_losses)
                all_time_wins += r
                all_time_losses += (1 - r)
                all_time_pct = all_time_wins/(all_time_wins + all_time_losses)
    
                aggregate.loc[len(aggregate)] = [owner, season, week, opp, pts_for, pts_ag, r, szn_pts_for, szn_pts_against,
                                                 szn_wins, szn_losses, szn_pct, all_time_pts_for, all_time_pts_against, all_time_wins,
                                                 all_time_losses, all_time_pct, reg_rank, avg_reg_rank, rank, avg_final_rank, championships]
    return aggregate




def summarize_by_owner(aggregate):
    end_of_szn = aggregate.loc[aggregate.groupby(['season', 'owner'])['week'].idxmax()]
    
    owner_summary = end_of_szn.groupby('owner').agg(
        total_wins=('szn_wins','sum'),
        total_losses=('szn_losses', 'sum'),
        total_pts_for=('szn_pts_for','sum'),
        total_pts_against=('szn_pts_against','sum'),
        avg_wins=('szn_wins', 'mean'),
        avg_losses=('szn_losses', 'mean'),
        avg_pts_for=('szn_pts_for', 'mean'),
        avg_pts_against=('szn_pts_against', 'mean'),
        avg_reg_szn_rank=('reg_szn_rank', 'mean'),
        avg_final_rank=('final_rank', 'mean'),
        win_percentage=("szn_pct", "mean"),
        total_championships=('championships', 'max')
    ).reset_index()
    return owner_summary


def summarize_by_owner_season(aggregate):
    end_of_szn = aggregate.loc[aggregate.groupby(['season', 'owner'])['week'].idxmax()]
    
    szn_summary = end_of_szn.groupby(['season', 'owner']).agg(
        avg_wins=('szn_wins', 'mean'),
        avg_losses=('szn_losses', 'mean'),
        avg_pts_for=('szn_pts_for', 'mean'),
        avg_pts_against=('szn_pts_against', 'mean'),
        avg_reg_szn_rank=('reg_szn_rank', 'mean'),
        avg_final_rank=('final_rank', 'mean'),
        win_percentage=('szn_pct', 'max'),
        total_championships=('championships', 'max')
    ).reset_index()
    return szn_summary


def head_to_head(clean_weekly):
    comp = pd.DataFrame(columns = ['owner', 'opponent', 'owner_wins', 'opp_wins', 'owner_pts', 'opp_pts'])
    for owner in clean_weekly['owner'].unique():
        own_ref = clean_weekly[clean_weekly['owner'] == owner]
        for opp in own_ref['opponent'].unique():
            h2h = own_ref[own_ref['opponent'] == opp]

            wins = sum(h2h['result'])
            losses = len(h2h['result']) - sum(h2h['result'])
            pts_for = sum(h2h['pts_for']) 
            pts_against = sum(h2h['pts_against'])
            comp.loc[len(comp)] = [owner, opp, wins, losses, pts_for, pts_against]
    return comp


def pull_playoffs(aggregate):
    playoffs = aggregate.iloc[:0].copy()

    for season in aggregate['season'].unique():
        szn_ref = aggregate[aggregate['season'] == season].sort_values('week')
        playoff_weeks = szn_ref['week'].unique()[-2:]  
        playoff_games = szn_ref[szn_ref['week'].isin(playoff_weeks)]
        playoffs = pd.concat([playoffs, playoff_games], ignore_index=True)
    return playoffs


def playoff_metrics(playoffs):
    playoff_metrics = pd.DataFrame(columns = ['owner', 'playoff_wins', 'playoff_pts_for', 'playoff_pts_against', 'championships', 'last_places', 'rank_differential'])
    for owner in playoffs.owner.unique():
        ref = playoffs[playoffs['owner'] == owner]
        lasts = len(ref[ref['final_rank'] == 8])/2
        ref = ref[ref['reg_szn_rank'] <= 4]
        
        wins = sum(ref['result'])
        pts = sum(ref['pts_for'])
        pts_against = sum(ref['pts_against'])
        champs = max(ref['championships'])
        
        diff = np.mean(ref['reg_szn_rank'] - np.mean(ref['final_rank'])) 
        
        
        playoff_metrics.loc[len(playoff_metrics)] = [owner, wins, pts, pts_against, champs, lasts, diff]
    return playoff_metrics
   

