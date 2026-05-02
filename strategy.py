#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakouts capture sustained moves; 1w EMA50 ensures alignment with weekly trend
# Volume spike (>2.0 x 20-period EMA) confirms breakout validity
# Works in bull markets (breakout above upper band + uptrend) and bear markets (breakdown below lower band + downtrend)
# Uses discrete position sizing (0.30) to balance return and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag

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
    
    # 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        window_high = high[i - lookback + 1:i + 1]
        window_low = low[i - lookback + 1:i + 1]
        highest_high[i] = np.max(window_high)
        lowest_low[i] = np.min(window_low)
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and EMA calculation)
    start_idx = max(lookback - 1, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above upper Donchian band with volume confirmation and uptrend
            if close[i] > highest_high[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below lower Donchian band with volume confirmation and downtrend
            elif close[i] < lowest_low[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price closes below midpoint of Donchian channel OR trend changes to downtrend
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] < midpoint or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price closes above midpoint of Donchian channel OR trend changes to uptrend
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] > midpoint or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals