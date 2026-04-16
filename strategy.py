#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w ATR filter
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND 1w ATR(14) < median
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND 1w ATR(14) < median
# Exit on opposite Donchian breakout (reverse signal)
# Donchian provides clear trend structure, volume confirms conviction, low weekly ATR filters for low-volatility breakouts
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Donchian channels (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Volume confirmation (20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 1w ATR filter (14-period) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Median ATR for filtering (use 50-period median)
    atr_median = pd.Series(atr_1w).rolling(window=50, min_periods=50).median().values
    atr_median_aligned = align_htf_to_ltf(prices, df_1w, atr_median)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(atr_median_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        atr_med_val = atr_median_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 1d average volume
        vol_confirm = volume[i] > vol_ma_val * 1.5
        
        # ATR filter: low volatility environment (current ATR < median)
        # Need current 1w ATR value
        atr_current = atr_1w[len(df_1w) - len(atr_1w_aligned) + i] if i >= len(atr_1w) - len(atr_1w_aligned) else 0
        # Simpler: use the ATR value aligned to current time
        # We'll compute aligned ATR
        atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
        atr_current = atr_1w_aligned[i]
        low_vol = atr_current < atr_med_val if not np.isnan(atr_current) and not np.isnan(atr_med_val) else False
        
        # === ENTRY LOGIC ===
        if position == 0:
            # Long when: price breaks above Donchian high AND volume confirmation AND low volatility
            if price > upper and vol_confirm and low_vol:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price breaks below Donchian low AND volume confirmation AND low volatility
            elif price < lower and vol_confirm and low_vol:
                signals[i] = -0.25
                position = -1
                continue
        
        # === EXIT LOGIC: reverse signal ===
        elif position == 1:
            # Exit long if price breaks below Donchian low
            if price < lower:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short if price breaks above Donchian high
            if price > upper:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolume1.5x_1wATRLowVol"
timeframe = "4h"
leverage = 1.0