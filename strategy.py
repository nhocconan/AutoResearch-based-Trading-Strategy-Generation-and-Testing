#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian breakout (20-period) with 1d EMA50 trend filter and volume confirmation (>1.5x average).
- Uses 4h for signal direction (Donchian breakout) and 1d for trend filter (EMA50)
- Volume confirmation reduces false breakouts
- Session filter: 08-20 UTC to avoid low-liquidity periods
- Position size: 0.20 (discrete level to minimize fee churn)
- Target: 60-150 total trades over 4 years = 15-37/year for 1h
- Works in bull/bear via trend filter and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 1.5x 24-period average (for 1h, 24h lookback)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian upper/lower bands
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (use prior completed 4h bar)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 24)  # EMA50, Donchian20, volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(upper_4h_aligned[i]) or
            np.isnan(lower_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout signals (using current close vs prior levels)
        breakout_up = close[i] > upper_4h_aligned[i-1]  # Close above prior 4h upper band
        breakout_down = close[i] < lower_4h_aligned[i-1]  # Close below prior 4h lower band
        
        if position == 0:
            # Long: 4h Donchian breakout up AND price > 1d EMA50 AND volume confirmation AND in session
            if breakout_up and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h Donchian breakout down AND price < 1d EMA50 AND volume confirmation AND in session
            elif breakout_down and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h Donchian break down OR price < 1d EMA50 (trend flip)
            if close[i] < lower_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h Donchian break up OR price > 1d EMA50 (trend flip)
            if close[i] > upper_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_Breakout_1dEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0