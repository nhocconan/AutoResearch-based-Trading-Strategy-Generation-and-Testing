#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_Volume
Hypothesis: Uses daily price action relative to weekly Camarilla pivot levels for breakout entries with volume confirmation.
Designed to capture multi-day institutional breakouts with volume confirmation, filtering by weekly trend via EMA20.
Works in bull/bear by following weekly trend direction - avoids counter-trend losses.
Targets 10-25 trades/year per symbol with focus on high-probability setups.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Pivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter and Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly volume average for confirmation
    volume_1w = df_1w['volume'].values
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    # Calculate weekly Camarilla levels: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    camarilla_r3_1w = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_s3_1w = close_1w - 1.1 * (high_1w - low_1w)
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    
    # Daily volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20_1w_aligned[i]) or 
            np.isnan(camarilla_r3_1w_aligned[i]) or np.isnan(camarilla_s3_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: daily volume > 1.5x 20-day average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Weekly volume confirmation: current week's volume > 1.3x 20-week average
        weekly_volume_filter = False
        # Find the index in the weekly aligned arrays for current week
        # We use the current value as proxy for weekly volume condition
        weekly_volume_filter = volume[i] > 1.3 * vol_avg_20_1w_aligned[i]
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Breakout conditions using weekly Camarilla levels
        breakout_up = close[i] > camarilla_r3_1w_aligned[i]  # Break above weekly R3
        breakdown_down = close[i] < camarilla_s3_1w_aligned[i]  # Break below weekly S3
        
        # Entry conditions: only trade in direction of weekly trend with volume confirmation
        long_entry = breakout_up and volume_filter and weekly_volume_filter and uptrend
        short_entry = breakdown_down and volume_filter and weekly_volume_filter and downtrend
        
        # Exit conditions: return to opposite weekly Camarilla level or trend reversal
        long_exit = (close[i] < camarilla_s3_1w_aligned[i]) or (not uptrend)  # Break below S3 or trend change
        short_exit = (close[i] > camarilla_r3_1w_aligned[i]) or (not downtrend)  # Break above R3 or trend change
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals