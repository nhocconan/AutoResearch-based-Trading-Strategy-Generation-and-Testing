#!/usr/bin/env python3
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
    
    # === 12h Williams Fractals (requires 2-bar confirmation) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Williams Fractal: bearish = high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]
    # bullish = low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]
    bearish_fractal = np.zeros_like(high_12h, dtype=bool)
    bullish_fractal = np.zeros_like(low_12h, dtype=bool)
    
    for i in range(2, len(high_12h)-2):
        if (high_12h[i] > high_12h[i-1] and high_12h[i] > high_12h[i-2] and 
            high_12h[i] > high_12h[i+1] and high_12h[i] > high_12h[i+2]):
            bearish_fractal[i] = True
        if (low_12h[i] < low_12h[i-1] and low_12h[i] < low_12h[i-2] and 
            low_12h[i] < low_12h[i+1] and low_12h[i] < low_12h[i+2]):
            bullish_fractal[i] = True
    
    # Requires 2 extra bars for confirmation (per rule 2b)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # === 6h Donchian Channel (20-period) ===
    donchian_len = 20
    highest_high = np.full_like(close, np.nan)
    lowest_low = np.full_like(close, np.nan)
    
    for i in range(n):
        if i >= donchian_len - 1:
            highest_high[i] = np.max(high[i-donchian_len+1:i+1])
            lowest_low[i] = np.min(low[i-donchian_len+1:i+1])
    
    # === 12h Volume Spike Confirmation ===
    vol_ma_20 = np.full_like(df_12h['volume'].values, np.nan)
    vol_12h = df_12h['volume'].values
    for i in range(len(vol_12h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(vol_12h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(vol_12h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = vol_12h[0]
    
    vol_spike = vol_12h > vol_ma_20 * 2.0  # 2x volume spike
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish fractal + price breaks above Donchian high + volume spike
            if (bullish_fractal_aligned[i] > 0.5 and 
                close[i] > highest_high[i] and 
                vol_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish fractal + price breaks below Donchian low + volume spike
            elif (bearish_fractal_aligned[i] > 0.5 and 
                  close[i] < lowest_low[i] and 
                  vol_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to Donchian low OR opposite fractal with volume
            if (close[i] < lowest_low[i] or 
                (bearish_fractal_aligned[i] > 0.5 and vol_spike_aligned[i] > 0.5)):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian high OR opposite fractal with volume
            if (close[i] > highest_high[i] or 
                (bullish_fractal_aligned[i] > 0.5 and vol_spike_aligned[i] > 0.5)):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Fractal_DonchianBreakout_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0