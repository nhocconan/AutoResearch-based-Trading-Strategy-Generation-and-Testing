#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(34) trend filter and volume spike confirmation.
# Works in bull markets by catching breakouts, in bear by only taking shorts below EMA34.
# Volume spike filters breakouts with conviction. Target: 20-40 trades/year.
def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian(20) channel
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 4h volume for spike detection
    vol_4h = df_4h['volume'].values
    vol_ma = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need all indicators
    start_idx = max(34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_val = ema_34_aligned[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        
        # Volume spike: current 4h volume > 1.5x 20-period average
        vol_spike = volume[i] > (1.5 * vol_ma_val)
        
        if position == 0:
            # Long: price breaks above upper Donchian, above EMA34, with volume spike
            if close[i] > upper_val and close[i] > ema_val and vol_spike:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian, below EMA34, with volume spike
            elif close[i] < lower_val and close[i] < ema_val and vol_spike:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower Donchian
            if close[i] < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above upper Donchian
            if close[i] > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_EMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0