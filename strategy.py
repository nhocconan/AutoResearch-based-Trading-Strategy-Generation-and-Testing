# 4h_Camarilla_R1_S1_Breakout_Volume_Regime_v1
# Camarilla pivot breakout strategy on 4h timeframe with volume confirmation and chop regime filter
# Uses daily Camarilla levels (R1, S1) for entry/exit, volume spike for confirmation, and Choppiness Index for regime filtering
# Designed to work in both bull and bear markets by combining mean reversion at pivot levels with trend following in strong trends
# Target: 75-200 total trades over 4 years (19-50/year)

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
    
    # === 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    
    # === 1d Choppiness Index (14-period) ===
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    atr_14 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 13:
            atr_14[i] = np.mean(tr[i-13:i+1])
        elif i > 0:
            atr_14[i] = np.mean(tr[1:i+1])
        else:
            atr_14[i] = np.nan
    
    # Calculate sum of True Range over 14 periods
    sum_tr_14 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 13:
            sum_tr_14[i] = np.sum(tr[i-13:i+1])
        elif i > 0:
            sum_tr_14[i] = np.sum(tr[1:i+1])
        else:
            sum_tr_14[i] = np.nan
    
    # Choppiness Index = 100 * log10(sum(tr14) / (atr14 * 14)) / log10(14)
    chop = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(sum_tr_14[i]) and not np.isnan(atr_14[i]) and atr_14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr_14[i] / (atr_14[i] * 14)) / np.log10(14)
        else:
            chop[i] = np.nan
    
    # === Align indicators to 4h timeframe ===
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Volume confirmation ===
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 in trending market (CHOP < 38.2) OR mean reversion to S1 in ranging market (CHOP > 61.8)
            if ((close[i] > r1_aligned[i] and chop_aligned[i] < 38.2) or  # breakout in trend
                (close[i] < s1_aligned[i] and chop_aligned[i] > 61.8)):   # mean reversion in range
                if vol_confirm[i]:  # volume confirmation
                    signals[i] = 0.25
                    position = 1
                    continue
            # Short: price breaks below S1 in trending market (CHOP < 38.2) OR mean reversion to R1 in ranging market (CHOP > 61.8)
            elif ((close[i] < s1_aligned[i] and chop_aligned[i] < 38.2) or  # breakdown in trend
                  (close[i] > r1_aligned[i] and chop_aligned[i] > 61.8)):   # mean reversion in range
                if vol_confirm[i]:  # volume confirmation
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to pivot OR opposite signal with volume
            pivot_aligned = align_htf_to_ltf(prices, df_1d, (high_1d + low_1d + close_1d) / 3)
            if (close[i] < pivot_aligned[i] and vol_confirm[i]) or \
               (close[i] < s1_aligned[i] and chop_aligned[i] > 61.8 and vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot OR opposite signal with volume
            pivot_aligned = align_htf_to_ltf(prices, df_1d, (high_1d + low_1d + close_1d) / 3)
            if (close[i] > pivot_aligned[i] and vol_confirm[i]) or \
               (close[i] > r1_aligned[i] and chop_aligned[i] > 61.8 and vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0