#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike.
# Donchian breakouts capture momentum in both bull and bear markets. 
# Entry on breakout of 20-period high/low, confirmed by 1w EMA50 trend direction and 1d volume spike (>1.5x 20-period MA).
# Exit on opposite Donchian breakout or volume dry-up. 
# Designed to work in trending markets (bull/bear) with filters to avoid whipsaws in ranging conditions.
# Target: 15-30 trades/year (60-120 total over 4 years).

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume 20-period MA for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (vol_ma_20[i] * 1.5)
        
        if position == 0:
            # Long: price breaks above 20-period high AND above 1w EMA50 AND volume spike AND session
            if close[i] > highest_high_20[i] and close[i] > ema_50_1w_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low AND below 1w EMA50 AND volume spike AND session
            elif close[i] < lowest_low_20[i] and close[i] < ema_50_1w_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-period low OR volume dry-up (< 0.5x MA)
            if close[i] < lowest_low_20[i] or volume[i] < (vol_ma_20[i] * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-period high OR volume dry-up (< 0.5x MA)
            if close[i] > highest_high_20[i] or volume[i] < (vol_ma_20[i] * 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals