#!/usr/bin/env python3
"""
4h_1d_Keltner_Channel_MeanReversion_v1
Hypothesis: On 4h timeframe, use 1d Keltner Channel (ATR-based) for mean reversion with volume confirmation and ADX trend filter.
Buy when price touches lower Keltner band with volume spike in a low-volatility regime (ADX < 25).
Sell when price touches upper Keltner band with volume spike in a low-volatility regime (ADX < 25).
Exit when price reverts to the 1d EMA20 (middle band) or ADX rises above 25 indicating trend.
Keltner Channels adapt to volatility better than fixed % bands, working in both ranging and trending markets.
Designed for 4h to capture reversals while avoiding false signals in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_ata, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Keltner Channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA20 (middle band)
    ema20_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        ema20_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema20_1d[i] = (close_1d[i] * 2/21) + (ema20_1d[i-1] * 19/21)
    
    # Calculate 1d ATR(10) for Keltner Channel width
    atr_1d = np.full_like(high_1d, np.nan)
    if len(high_1d) >= 11:
        tr = np.zeros_like(high_1d)
        tr[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(high_1d)):
            tr[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - high_1d[i-1]), 
                       abs(low_1d[i] - low_1d[i-1]))
        atr_1d[10] = np.mean(tr[1:11])
        for i in range(11, len(high_1d)):
            atr_1d[i] = (tr[i] * 2/11) + (atr_1d[i-1] * 9/10)
    
    # Calculate Keltner Bands: EMA20 ± 2 * ATR(10)
    keltner_lower = np.full_like(close_1d, np.nan)
    keltner_upper = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if not (np.isnan(ema20_1d[i]) or np.isnan(atr_1d[i])):
            keltner_lower[i] = ema20_1d[i] - 2 * atr_1d[i]
            keltner_upper[i] = ema20_1d[i] + 2 * atr_1d[i]
    
    # Calculate ADX on 1d data
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr_adx = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]):
            continue
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr_adx[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - high_1d[i-1]), 
                       abs(low_1d[i] - low_1d[i-1]))
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/14)
    atr_adx = np.zeros_like(high_1d)
    plus_di = np.zeros_like(high_1d)
    minus_di = np.zeros_like(high_1d)
    dx = np.zeros_like(high_1d)
    adx = np.full_like(high_1d, np.nan)
    
    if len(high_1d) >= 14:
        # Initial values
        atr_adx[13] = np.nansum(tr_adx[1:14])
        plus_dm_sum = np.nansum(plus_dm[1:14])
        minus_dm_sum = np.nansum(minus_dm[1:14])
        
        for i in range(14, len(high_1d)):
            if np.isnan(tr_adx[i]) or np.isnan(plus_dm[i]) or np.isnan(minus_dm[i]):
                atr_adx[i] = atr_adx[i-1]
                plus_dm_sum = plus_dm_sum
                minus_dm_sum = minus_dm_sum
            else:
                atr_adx[i] = (atr_adx[i-1] * 13 + tr_adx[i]) / 14
                plus_dm_sum = (plus_dm_sum * 13 + plus_dm[i]) / 14
                minus_dm_sum = (minus_dm_sum * 13 + minus_dm[i]) / 14
            
            if atr_adx[i] > 0:
                plus_di[i] = 100 * plus_dm_sum / atr_adx[i]
                minus_di[i] = 100 * minus_dm_sum / atr_adx[i]
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # Calculate ADX as smoothed DX
        if len(high_1d) >= 27:
            adx[26] = np.nanmean(dx[14:27])
            for i in range(27, len(high_1d)):
                if np.isnan(dx[i]):
                    adx[i] = adx[i-1]
                else:
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align Keltner levels and ADX to 4h timeframe
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # Start after enough data for alignment
        # Skip if any critical data is NaN
        if (np.isnan(keltner_lower_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        vol_ma_20 = np.full_like(volume, np.nan)
        for j in range(19, len(volume)):
            vol_ma_20[j] = np.mean(volume[j-19:j+1])
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Look for long entries: price touches lower Keltner band with volume spike in low volatility regime
            if (close[i] <= keltner_lower_aligned[i] * 1.001 and  # Allow small tolerance
                volume_ratio > 1.8 and
                adx_aligned[i] < 25):
                position = 1
                signals[i] = position_size
            # Look for short entries: price touches upper Keltner band with volume spike in low volatility regime
            elif (close[i] >= keltner_upper_aligned[i] * 0.999 and  # Allow small tolerance
                  volume_ratio > 1.8 and
                  adx_aligned[i] < 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches middle band (EMA20) or ADX rises indicating trend
            if (close[i] >= ema20_1d_aligned[i] * 0.999 or
                adx_aligned[i] >= 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches middle band (EMA20) or ADX rises indicating trend
            if (close[i] <= ema20_1d_aligned[i] * 1.001 or
                adx_aligned[i] >= 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Keltner_Channel_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0