#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1D DATA FOR CAMARILLA PIVOTS AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + 1.1 * Range / 2
    # S3 = Pivot - 1.1 * Range / 2
    # R4 = Pivot + 1.1 * Range
    # S4 = Pivot - 1.1 * Range
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + 1.1 * range_1d / 2
    s3_1d = pivot_1d - 1.1 * range_1d / 2
    r4_1d = pivot_1d + 1.1 * range_1d
    s4_1d = pivot_1d - 1.1 * range_1d
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1D EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema34_1d_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume, continue if above R4
            if (close[i] > r3_6h[i] and volume_spike[i]):
                if close[i] > r4_6h[i]:
                    # Strong breakout - full position
                    signals[i] = 0.30
                else:
                    # Moderate breakout - half position
                    signals[i] = 0.15
                position = 1
            # SHORT: Break below S3 with volume, continue if below S4
            elif (close[i] < s3_6h[i] and volume_spike[i]):
                if close[i] < s4_6h[i]:
                    # Strong breakdown - full position
                    signals[i] = -0.30
                else:
                    # Moderate breakdown - half position
                    signals[i] = -0.15
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (reversal signal)
            if close[i] < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if close[i] > r4_6h[i] else 0.15
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (reversal signal)
            if close[i] > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30 if close[i] < s4_6h[i] else -0.15
    
    return signals