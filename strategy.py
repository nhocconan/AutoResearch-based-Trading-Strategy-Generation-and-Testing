#!/usr/bin/env python3
"""
4h_1d_TRIX_Trend_Follower_v1
Hypothesis: TRIX (15) on 1d timeframe filters trend direction on 4h. Long when TRIX > 0 and rising, short when TRIX < 0 and falling.
Volume confirmation: current 4h volume > 1.5x 20-period average. Entry only in strong trend (ADX > 25 on 1d).
Exit when TRIX crosses zero or volume drops below average.
Designed to capture medium-term trends in both bull and bear markets while avoiding whipsaws in ranging conditions.
"""

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
    
    # Load 1d data for TRIX and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TRIX on 1d close (15-period EMA of EMA of EMA, then percent change)
    def ema(array, period):
        result = np.full_like(array, np.nan, dtype=float)
        if len(array) < period:
            return result
        alpha = 2 / (period + 1)
        result[period-1] = np.nanmean(array[:period])
        for i in range(period, len(array)):
            if np.isnan(array[i]):
                result[i] = result[i-1]
            else:
                result[i] = alpha * array[i] + (1 - alpha) * result[i-1]
        return result
    
    ema1 = ema(close_1d, 15)
    ema2 = ema(ema1, 15)
    ema3 = ema(ema2, 15)
    
    trix = np.full_like(close_1d, np.nan)
    for i in range(1, len(ema3)):
        if np.isnan(ema3[i]) or np.isnan(ema3[i-1]) or ema3[i-1] == 0:
            trix[i] = np.nan
        else:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Calculate ADX on 1d data
    if len(high_1d) < 14:
        return np.zeros(n)
    
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]):
            continue
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - high_1d[i-1]), 
                   abs(low_1d[i] - low_1d[i-1]))
    
    # Wilder's smoothing
    atr = np.zeros_like(high_1d)
    plus_di = np.zeros_like(high_1d)
    minus_di = np.zeros_like(high_1d)
    dx = np.zeros_like(high_1d)
    adx = np.full_like(high_1d, np.nan)
    
    if len(high_1d) >= 14:
        atr[13] = np.nansum(tr[1:14])
        plus_dm_sum = np.nansum(plus_dm[1:14])
        minus_dm_sum = np.nansum(minus_dm[1:14])
        
        for i in range(14, len(high_1d)):
            if np.isnan(tr[i]) or np.isnan(plus_dm[i]) or np.isnan(minus_dm[i]):
                atr[i] = atr[i-1]
                plus_dm_sum = plus_dm_sum
                minus_dm_sum = minus_dm_sum
            else:
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
                plus_dm_sum = (plus_dm_sum * 13 + plus_dm[i]) / 14
                minus_dm_sum = (minus_dm_sum * 13 + minus_dm[i]) / 14
            
            if atr[i] > 0:
                plus_di[i] = 100 * plus_dm_sum / atr[i]
                minus_di[i] = 100 * minus_dm_sum / atr[i]
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        if len(high_1d) >= 27:
            adx[26] = np.nanmean(dx[14:27])
            for i in range(27, len(high_1d)):
                if np.isnan(dx[i]):
                    adx[i] = adx[i-1]
                else:
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align TRIX and ADX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume moving average (20-period)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX > 0 and rising (current > previous), volume confirmation, strong trend
            if (trix_aligned[i] > 0 and 
                trix_aligned[i] > trix_aligned[i-1] and
                volume[i] > 1.5 * vol_ma_20[i] and
                adx_aligned[i] > 25):
                position = 1
                signals[i] = position_size
            # Short: TRIX < 0 and falling (current < previous), volume confirmation, strong trend
            elif (trix_aligned[i] < 0 and 
                  trix_aligned[i] < trix_aligned[i-1] and
                  volume[i] > 1.5 * vol_ma_20[i] and
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TRIX crosses below zero or volume drops below average
            if (trix_aligned[i] <= 0 or
                volume[i] <= vol_ma_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: TRIX crosses above zero or volume drops below average
            if (trix_aligned[i] >= 0 or
                volume[i] <= vol_ma_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_TRIX_Trend_Follower_v1"
timeframe = "4h"
leverage = 1.0