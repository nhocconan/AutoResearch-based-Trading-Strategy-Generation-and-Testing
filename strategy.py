#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with volume spike and weekly trend filter.
Long when price breaks above 20-day high AND 1d volume > 2.0x 20-bar average AND weekly close > weekly EMA34.
Short when price breaks below 20-day low AND 1d volume > 2.0x 20-bar average AND weekly close < weekly EMA34.
Exit when price touches 10-day EMA or opposite Donchian level.
Uses 1d for execution and Donchian levels, 1w for trend filter.
Designed to capture strong momentum moves with volume confirmation in the direction of the weekly trend.
Target: 15-25 trades/year per symbol.
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
    
    # Get 1d data for Donchian levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    high_roll_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume MA for confirmation
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d 10-period EMA for exit
    ema_10 = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_roll_max)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_roll_min)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(ema_10_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-bar average
        volume_confirmed = volume_1d[i] > 2.0 * vol_ma_20_aligned[i]
        
        # Trend filter: weekly close above/below weekly EMA34
        weekly_uptrend = close_1w[i // 7] > ema_34_1w[i // 7] if i // 7 < len(close_1w) else False
        weekly_downtrend = close_1w[i // 7] < ema_34_1w[i // 7] if i // 7 < len(close_1w) else False
        
        # Breakout conditions
        breakout_high = close_1d[i] > donchian_high_aligned[i]
        breakout_low = close_1d[i] < donchian_low_aligned[i]
        
        # Exit conditions: touch 10-day EMA or opposite Donchian level
        touch_ema = abs(close_1d[i] - ema_10_aligned[i]) < 0.005 * close_1d[i]  # within 0.5%
        touch_opposite = (position == 1 and close_1d[i] < donchian_low_aligned[i]) or \
                         (position == -1 and close_1d[i] > donchian_high_aligned[i])
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and weekly uptrend
            if (breakout_high and volume_confirmed and weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and weekly downtrend
            elif (breakout_low and volume_confirmed and weekly_downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch 10-day EMA or break below Donchian low
            if (touch_ema or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch 10-day EMA or break above Donchian high
            if (touch_ema or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0