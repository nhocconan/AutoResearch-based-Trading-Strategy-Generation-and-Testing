#!/usr/bin/env python3
"""
1d_FundingRate_MeanReversion_WeeklyTrend_Filter
Hypothesis: Funding rate mean-reversion on 1d timeframe filtered by weekly EMA trend.
Enter long when 30d funding rate z-score < -2.0 (extreme pessimism) and weekly uptrend.
Enter short when 30d funding rate z-score > +2.0 (extreme optimism) and weekly downtrend.
Exit when z-score reverts toward zero (|z| < 0.5) or opposite extreme.
Designed for low trade frequency (target: 10-20 trades/year) to minimize fee drag.
Works in bull/bear via weekly trend alignment and funding extremes as contrarian signal.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load funding rate data (assuming available as parquet, same structure as prices)
    try:
        funding_df = pd.read_parquet('data/processed/funding/BTCUSDT.parquet')
    except:
        # Fallback: if funding data not available, use zero signal
        return np.zeros(n)
    
    # Ensure we have funding rate column
    if 'funding_rate' not in funding_df.columns:
        return np.zeros(n)
    
    # Align funding data to prices timeframe (1d)
    funding_rates = funding_df['funding_rate'].values
    funding_times = funding_df['open_time'].values
    
    # Create funding series aligned to prices index
    funding_aligned = np.full(n, np.nan)
    # Simple alignment: find closest funding rate for each price bar
    for i in range(n):
        price_time = prices['open_time'].iloc[i]
        # Find funding rate from same day (simplified)
        mask = funding_times <= price_time
        if np.any(mask):
            funding_aligned[i] = funding_rates[mask][-1]
    
    # Calculate 30-day z-score of funding rate
    funding_series = pd.Series(funding_aligned)
    funding_ma = funding_series.rolling(window=30, min_periods=30).mean().values
    funding_std = funding_series.rolling(window=30, min_periods=30).std().values
    funding_z = (funding_aligned - funding_ma) / funding_std
    # Replace infinite/NaN values
    funding_z = np.nan_to_num(funding_z, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Load HTF data ONCE before loop (weekly for trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === Weekly EMA34 for HTF trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(funding_z[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        z = funding_z[i]
        price = prices['close'].iloc[i]
        ema_trend = ema_34_1w_aligned[i]
        
        if position == 0:
            # Entry conditions
            long_signal = (z < -2.0) and (price > ema_trend)  # Extreme pessimism + weekly uptrend
            short_signal = (z > 2.0) and (price < ema_trend)  # Extreme optimism + weekly downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: z-score reverts or extreme optimism
            if (z > -0.5) or (z > 2.0):  # Reversion to neutral or opposite extreme
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: z-score reverts or extreme pessimism
            if (z < 0.5) or (z < -2.0):  # Reversion to neutral or opposite extreme
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_FundingRate_MeanReversion_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0