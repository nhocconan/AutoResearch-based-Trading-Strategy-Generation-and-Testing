#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day high AND close > 1w EMA50 (uptrend) AND volume spike
# Short when price breaks below 20-day low AND close < 1w EMA50 (downtrend) AND volume spike
# Uses Donchian channels for structure, 1w EMA50 for higher-timeframe trend filter (avoid counter-trend),
# volume spike for conviction. Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation).
# Timeframe: 1d (slower timeframe reduces trade frequency, lowers fee drag).

name = "1d_Donchian20_1wEMA50_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels from previous 1d bar (completed daily bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Rolling window of 20 completed daily bars for Donchian
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed daily bars (look-ahead safety)
    high_20_shifted = np.roll(high_20, 1)
    low_20_shifted = np.roll(low_20, 1)
    
    # Align Donchian levels to 1d timeframe (already aligned, but keep for consistency)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20_shifted)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20_shifted)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Higher threshold for fewer trades
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high AND uptrend (price > 1w EMA50) AND volume spike
            if (close[i] > high_20_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND downtrend (price < 1w EMA50) AND volume spike
            elif (close[i] < low_20_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 20-day high OR closes below 1w EMA50
            if close[i] < high_20_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 20-day low OR closes above 1w EMA50
            if close[i] > low_20_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals