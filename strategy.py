#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend and pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    range_hl = high_1d[:-1] - low_1d[:-1]
    
    # R3 and S3 levels (fade zone)
    r3 = pivot + range_hl * 1.1
    s3 = pivot - range_hl * 1.1
    
    # R4 and S4 levels (breakout zone)
    r4 = pivot + range_hl * 1.5
    s4 = pivot - range_hl * 1.5
    
    # Align pivot levels to 6h timeframe (shift by 1 to avoid look-ahead)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], pivot]))
    r3_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], r3]))
    s3_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], s3]))
    r4_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], r4]))
    s4_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[np.nan], s4]))
    
    # Volume confirmation - 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # ATR filter for volatility regime
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Fade at R3/S3: price touches extreme level with volume confirmation
            if (abs(price - r3_aligned[i]) < 0.001 * price and  # Near R3
                vol > 1.5 * vol_ma_20_aligned[i] and
                atr_14_aligned[i] > 0):
                signals[i] = -0.25  # Short at R3
                position = -1
            elif (abs(price - s3_aligned[i]) < 0.001 * price and  # Near S3
                  vol > 1.5 * vol_ma_20_aligned[i] and
                  atr_14_aligned[i] > 0):
                signals[i] = 0.25   # Long at S3
                position = 1
            # Breakout at R4/S4: price breaks beyond extreme level with volume
            elif (price > r4_aligned[i] and
                  vol > 2.0 * vol_ma_20_aligned[i] and
                  atr_14_aligned[i] > 0):
                signals[i] = 0.25   # Long breakout
                position = 1
            elif (price < s4_aligned[i] and
                  vol > 2.0 * vol_ma_20_aligned[i] and
                  atr_14_aligned[i] > 0):
                signals[i] = -0.25  # Short breakout
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot or volatility drops
            if (abs(price - pivot_aligned[i]) < 0.001 * price or
                vol < 0.5 * vol_ma_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot or volatility drops
            if (abs(price - pivot_aligned[i]) < 0.001 * price or
                vol < 0.5 * vol_ma_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S4_FadeBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0