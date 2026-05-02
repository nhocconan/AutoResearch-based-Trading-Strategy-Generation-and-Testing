#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian channels provide clear breakout levels with institutional relevance
# 1d EMA34 provides higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>2.0 x 20-period EMA) confirms breakout validity
# Works in bull markets (break above upper Donchian + 1d EMA34 up) and bear markets (break below lower Donchian + 1d EMA34 down)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    # 1d data for Donchian calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels from previous 1d bar
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    lookback = 20
    prev_high_1d = df_1d['high'].shift(1).rolling(window=lookback, min_periods=lookback).max().values
    prev_low_1d = df_1d['low'].shift(1).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align Donchian levels to 12h timeframe (wait for completed 1d bar)
    upper_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and EMA calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above upper Donchian with volume confirmation and uptrend
            if close[i] > upper_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Donchian with volume confirmation and downtrend
            elif close[i] < lower_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below lower Donchian (reversal) OR trend changes to downtrend
            if close[i] < lower_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above upper Donchian (reversal) OR trend changes to uptrend
            if close[i] > upper_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals