#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike
# Long when: Price breaks above 20-period Donchian high AND 1w EMA50 is rising AND 1d volume > 1.8x 20-period average
# Short when: Price breaks below 20-period Donchian low AND 1w EMA50 is falling AND 1d volume > 1.8x 20-period average
# Exit when price touches opposite Donchian level (20-period low for long exit, high for short exit)
# Donchian provides clear structure with proven edge in SOL/ETH
# 1w EMA50 filters for major trend alignment (works in both bull/bear via trend direction)
# Volume spike confirms institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_DonchianBreakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Rising/falling EMA: compare current vs previous value
    ema_rising_1w = np.gradient(ema_50_1w) > 0  # Positive slope = rising
    ema_falling_1w = np.gradient(ema_50_1w) < 0  # Negative slope = falling
    
    # Align 1w EMA trend to 12h
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising_1w)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling_1w)
    
    # Get 1d data ONCE before loop for volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume MA
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike (current volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.8 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Calculate 12h Donchian channels (20-period)
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_spike_aligned[i])
        
        if position == 0:
            # Long: Break above Donchian high with rising 1w EMA and volume spike
            if close[i] > donchian_high[i] and ema_rising_aligned[i] and vol_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with falling 1w EMA and volume spike
            elif close[i] < donchian_low[i] and ema_falling_aligned[i] and vol_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: touch Donchian low (opposite side)
            if close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: touch Donchian high (opposite side)
            if close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals