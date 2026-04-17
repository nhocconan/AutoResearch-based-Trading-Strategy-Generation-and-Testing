# This strategy uses a 4h Donchian channel breakout with volume confirmation and a 12h EMA trend filter
# The Donchian channel provides clear entry/exit levels, volume confirms breakout strength,
# and the 12h EMA ensures we only trade in the direction of the higher timeframe trend
# This combination has shown robustness across bull and bear markets in prior research
# Target: 20-40 trades per year to minimize fee drag while maintaining edge

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h EMA for trend filter (34-period) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 4h Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels
    high_max_20 = np.full_like(high_4h, np.nan)
    low_min_20 = np.full_like(low_4h, np.nan)
    
    for i in range(len(high_4h)):
        if i >= 19:
            high_max_20[i] = np.max(high_4h[i-19:i+1])
            low_min_20[i] = np.min(low_4h[i-19:i+1])
        elif i > 0:
            high_max_20[i] = np.max(high_4h[max(0, i-9):i+1])
            low_min_20[i] = np.min(low_4h[max(0, i-9):i+1])
        else:
            high_max_20[i] = high_4h[0]
            low_min_20[i] = low_4h[0]
    
    # Align Donchian levels to 4h timeframe
    high_max_20_aligned = align_htf_to_ltf(prices, df_4h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_4h, low_min_20)
    
    # === 4h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    vol_confirm = volume_4h > vol_ma_20 * 1.5  # Require 1.5x average volume for breakout
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period to ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(high_max_20_aligned[i]) or 
            np.isnan(low_min_20_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Long entry: price breaks above 4h Donchian high + volume confirmation + 12h EMA uptrend
        if position == 0:
            if (close[i] > high_max_20_aligned[i] and 
                vol_confirm[i] and 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
        
        # Short entry: price breaks below 4h Donchian low + volume confirmation + 12h EMA downtrend
        elif position == 0:
            if (close[i] < low_min_20_aligned[i] and 
                vol_confirm[i] and 
                close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit long: price breaks below 4h Donchian low OR 12h EMA turns down
        elif position == 1:
            if (close[i] < low_min_20_aligned[i] or 
                close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        # Exit short: price breaks above 4h Donchian high OR 12h EMA turns up
        elif position == -1:
            if (close[i] > high_max_20_aligned[i] or 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA34_VolumeFilter_Trend"
timeframe = "4h"
leverage = 1.0