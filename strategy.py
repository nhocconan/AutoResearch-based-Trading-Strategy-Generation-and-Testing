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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period) - Wilder's smoothing
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily ADX (14-period) - Wilder's smoothing
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_14 = tr
    plus_di_14 = np.full(len(df_1d), np.nan)
    minus_di_14 = np.full(len(df_1d), np.nan)
    dx_14 = np.full(len(df_1d), np.nan)
    
    if len(df_1d) >= 14:
        # Smooth +DM, -DM, TR
        plus_dm_smooth = np.full(len(df_1d), np.nan)
        minus_dm_smooth = np.full(len(df_1d), np.nan)
        tr_smooth = np.full(len(df_1d), np.nan)
        
        plus_dm_smooth[13] = np.sum(plus_dm[1:15])
        minus_dm_smooth[13] = np.sum(minus_dm[1:15])
        tr_smooth[13] = np.sum(tr[1:15])
        
        for i in range(14, len(df_1d)):
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / 14) + tr[i]
        
        plus_di_14 = 100 * plus_dm_smooth / tr_smooth
        minus_di_14 = 100 * minus_dm_smooth / tr_smooth
        dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    
    adx_14 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 27:  # Need 14 + 14 for smoothing
        adx_14[26] = np.mean(dx_14[14:28])
        for i in range(27, len(df_1d)):
            adx_14[i] = (adx_14[i-1] * 13 + dx_14[i]) / 14
    
    # Align indicators to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_12h = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 12-hour Donchian channels (10-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 10:
        for i in range(9, n):
            donch_high[i] = np.max(high[i-9:i+1])
            donch_low[i] = np.min(low[i-9:i+1])
    
    # Calculate 12-hour volume moving average (10-period)
    volume_ma = np.full(n, np.nan)
    if n >= 10:
        for i in range(9, n):
            volume_ma[i] = np.mean(volume[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_12h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(adx_12h[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_12h[i] / close[i] < 0.003:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 70% of 10-period MA)
        if volume[i] < 0.7 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Skip low trend strength (ADX < 25)
        if adx_12h[i] < 25:
            signals[i] = 0.0
            continue
        
        # Calculate pivot levels based on previous day's range
        prev_high = high_1d[i-1] if i > 0 else high_1d[0]
        prev_low = low_1d[i-1] if i > 0 else low_1d[0]
        prev_close = close_1d[i-1] if i > 0 else close_1d[0]
        prev_range = prev_high - prev_low
        
        # Pivot levels for reversal at extremes
        r3 = prev_close + (prev_range * 1.1 / 4)  # Resistance 3
        s3 = prev_close - (prev_range * 1.1 / 4)  # Support 3
        r4 = prev_close + (prev_range * 1.1 / 2)  # Resistance 4
        s4 = prev_close - (prev_range * 1.1 / 2)  # Support 4
        
        # Align to 12h timeframe
        r3_12h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r3))[i]
        s3_12h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s3))[i]
        r4_12h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r4))[i]
        s4_12h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s4))[i]
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high AND above S3 (support hold) AND volume > 1.5x MA
            if close[i] > donch_high[i] and close[i] > s3_12h and volume[i] > 1.5 * volume_ma[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 12h Donchian low AND below R3 (resistance hold) AND volume > 1.5x MA
            elif close[i] < donch_low[i] and close[i] < r3_12h and volume[i] > 1.5 * volume_ma[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 12h Donchian low OR below S4
            if close[i] < donch_low[i] or close[i] < s4_12h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 12h Donchian high OR above R4
            if close[i] > donch_high[i] or close[i] > r4_12h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Pivot_S3R3_Donchian10_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0