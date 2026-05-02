#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian(20) provides clear structure: upper/lower bands from 20-day high/low
# 1w EMA50 gives higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>2.0 x 20-period EMA) confirms breakout validity
# Works in bull markets (price > upper band + 1w EMA50 up) and bear markets (price < lower band + 1w EMA50 down)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag
# Timeframe: 1d (primary), HTF: 1w

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA20 for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1w data for trend filter (EMA50) and Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w Donchian(20) calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_confirmation = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper with volume confirmation and uptrend
            if close[i] > donchian_upper_aligned[i] and volume_confirmation and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume confirmation and downtrend
            elif close[i] < donchian_lower_aligned[i] and volume_confirmation and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price closes below Donchian lower OR trend changes to downtrend
            if close[i] < donchian_lower_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price closes above Donchian upper OR trend changes to uptrend
            if close[i] > donchian_upper_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals