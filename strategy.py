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
    
    # === 1d Williams %R (14-period) for mean reversion ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    period = 14
    for i in range(len(high_1d)):
        if i >= period - 1:
            highest_high[i] = np.max(high_1d[i - period + 1:i + 1])
            lowest_low[i] = np.min(low_1d[i - period + 1:i + 1])
    
    williams_r = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(highest_high[i]) and not np.isnan(lowest_low[i]):
            denominator = highest_high[i] - lowest_low[i]
            if denominator != 0:
                williams_r[i] = ((highest_high[i] - close_1d[i]) / denominator) * -100
            else:
                williams_r[i] = -50  # neutral when range is zero
    
    # === 1d ATR (14-period) for volatility filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # === 12h Volume confirmation ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period average volume on 12h timeframe
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(len(volume_12h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_12h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_12h[0]
    
    # Volume confirmation: current 12h volume > 1.5x 20-period average
    vol_confirm = volume_12h > vol_ma_20 * 1.5
    
    # Align all indicators to main timeframe (12h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_12h, vol_confirm)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: Williams %R < -80 (oversold) + volatility filter + volume confirmation
            if (williams_r_aligned[i] < -80 and 
                atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                vol_confirm_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Williams %R > -20 (overbought) + volatility filter + volume confirmation
            elif (williams_r_aligned[i] > -20 and 
                  atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                  vol_confirm_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought)
            if williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold)
            if williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Volume_Confirm_VolatilityFilter_v1"
timeframe = "12h"
leverage = 1.0