# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with 1-day volume confirmation and 1-day ATR volatility filter
# Donchian(20) breakouts capture trending moves; volume confirmation ensures conviction;
# ATR filter avoids low-volatility chop. Target: 15-25 trades/year for low fee decay.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Donchian Channel (20-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: 20-period high
    donch_high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donch_low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donch_high_20_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_20_12h)
    donch_low_20_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_20_12h)
    
    # === 1-day Volume Spike (vs 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1-day ATR (14-period) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_20_12h_aligned[i]) or np.isnan(donch_low_20_12h_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(atr_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 12h price and volume (avoid calling get_htf_data in loop)
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['close'].values)
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['volume'].values)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume spike: current 1d volume > 1.5x 20-period average
        vol_spike = volume_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # Volatility filter: current ATR > 20-period average ATR (avoid low volatility chop)
        vol_filter = atr_1d[i] > atr_ma_20_1d_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close_12h_aligned[i] > donch_high_20_12h_aligned[i]
        breakout_down = close_12h_aligned[i] < donch_low_20_12h_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            if vol_spike and vol_filter:
                if breakout_up:
                    signals[i] = 0.25
                    position = 1
                    continue
                elif breakout_down:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic: exit when price returns to middle of channel or conditions fail
        elif position == 1:
            # Exit long if price returns to midpoint or volatility drops
            midpoint = (donch_high_20_12h_aligned[i] + donch_low_20_12h_aligned[i]) / 2
            if close_12h_aligned[i] < midpoint or not vol_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns to midpoint or volatility drops
            midpoint = (donch_high_20_12h_aligned[i] + donch_low_20_12h_aligned[i]) / 2
            if close_12h_aligned[i] > midpoint or not vol_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolatilityFilter_VolumeSpike"
timeframe = "12h"
leverage = 1.0