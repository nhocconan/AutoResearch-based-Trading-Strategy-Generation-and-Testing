#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_Session
Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (price > 4h EMA34 for long, < for short) and volume spike confirmation.
Trades only during 08-20 UTC to avoid low-liquidity hours. Uses 4h EMA34 for higher timeframe trend alignment to avoid counter-trend trades.
Volume spike confirms institutional participation. Camarilla levels provide mathematically derived support/resistance.
Designed for 1h timeframe to target 15-37 trades/year (60-150 total over 4 years) by using tight entry conditions.
Works in bull/bear markets by trading with the 4h trend and using volume to filter false breakouts.
Session filter reduces noise trades during off-hours.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate Camarilla levels from previous 1d bar
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R1 = close + 1.0*(high-low)/12 * 11? Wait, standard Camarilla:
    # Actually: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # But we want R1/S1: R1 = close + 1.1*(high-low)/12 * 11? No.
    # Correct Camarilla: 
    # H-L = high-low
    # R1 = close + H-L * 1.1/12
    # R2 = close + H-L * 1.1/6
    # R3 = close + H-L * 1.1/4
    # S1 = close - H-L * 1.1/12
    # S2 = close - H-L * 1.1/6
    # S3 = close - H-L * 1.1/4
    # However, many traders use R3/S3 for breakouts. Let's use R1/S1 as tighter levels.
    # R1 = close + (high-low) * 1.1 / 12
    # S1 = close - (high-low) * 1.1 / 12
    hl_1d = high_1d - low_1d
    camarilla_r1 = close_1d + hl_1d * 1.1 / 12
    camarilla_s1 = close_1d - hl_1d * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: volume > 2.0 * 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 4h EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0  # Force flat outside session
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r1_aligned[i]
        breakout_short = close[i] < camarilla_s1_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 AND close > 4h EMA34 AND volume spike
            if breakout_long and close[i] > ema34_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: breakout below S1 AND close < 4h EMA34 AND volume spike
            elif breakout_short and close[i] < ema34_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: breakout below S1 OR close < 4h EMA34 (trend change)
            if breakout_short or close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: breakout above R1 OR close > 4h EMA34 (trend change)
            if breakout_long or close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0