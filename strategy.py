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
    
    # Load 1d data for ATR and pivot points (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:  # Need enough for ATR14
        return np.zeros(n)
    
    # Previous day's high, low, close for standard pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points (standard)
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # 1d ATR(14) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid choppy markets)
        atr_ma_50 = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr_14_aligned[i] > atr_ma_50[i] if not np.isnan(atr_ma_50[i]) else False
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike AND volatility filter
            if (close[i] > r1_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike AND volatility filter
            elif (close[i] < s1_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite level (S1 for long, R1 for short)
            if position == 1:
                # Exit long: Price closes below S1
                if close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above R1
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Pivot_R1_S1_Breakout_Volume_VolatilityFilter"
timeframe = "6h"
leverage = 1.0