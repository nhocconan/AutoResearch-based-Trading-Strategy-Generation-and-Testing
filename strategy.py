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
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR (14-period) - Wilder's smoothing
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    low_close = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Calculate weekly ADX (14-period) - Wilder's smoothing
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_14 = tr
    plus_dm_smooth = np.full(len(df_1w), np.nan)
    minus_dm_smooth = np.full(len(df_1w), np.nan)
    tr_smooth = np.full(len(df_1w), np.nan)
    
    if len(df_1w) >= 14:
        plus_dm_smooth[13] = np.sum(plus_dm[1:15])
        minus_dm_smooth[13] = np.sum(minus_dm[1:15])
        tr_smooth[13] = np.sum(tr[1:15])
        
        for i in range(14, len(df_1w)):
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / 14) + tr[i]
        
        plus_di_14 = 100 * plus_dm_smooth / tr_smooth
        minus_di_14 = 100 * minus_dm_smooth / tr_smooth
        dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    else:
        plus_di_14 = np.full(len(df_1w), np.nan)
        minus_di_14 = np.full(len(df_1w), np.nan)
        dx_14 = np.full(len(df_1w), np.nan)
    
    adx_14 = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 27:
        adx_14[26] = np.mean(dx_14[14:28])
        for i in range(27, len(df_1w)):
            adx_14[i] = (adx_14[i-1] * 13 + dx_14[i]) / 14
    
    # Align weekly indicators to 6h timeframe
    atr_6h = align_htf_to_ltf(prices, df_1w, atr_1w)
    adx_6h = align_htf_to_ltf(prices, df_1w, adx_14)
    
    # Calculate 6h Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 6h volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_6h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(adx_6h[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_6h[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 65% of 20-period MA)
        if volume[i] < 0.65 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Skip low trend strength (ADX < 28)
        if adx_6h[i] < 28:
            signals[i] = 0.0
            continue
        
        # Calculate weekly pivot levels based on previous week's range
        prev_high = high_1w[i-1] if i > 0 else high_1w[0]
        prev_low = low_1w[i-1] if i > 0 else low_1w[0]
        prev_close = close_1w[i-1] if i > 0 else close_1w[0]
        prev_range = prev_high - prev_low
        
        # Pivot levels for reversal at extremes
        s3 = prev_close - (prev_range * 1.1 / 4)  # Support 3
        r3 = prev_close + (prev_range * 1.1 / 4)  # Resistance 3
        
        # Align to 6h timeframe
        s3_6h = align_htf_to_ltf(prices, df_1w, np.full(len(df_1w), s3))[i]
        r3_6h = align_htf_to_ltf(prices, df_1w, np.full(len(df_1w), r3))[i]
        
        if position == 0:
            # Long: Price breaks above 6h Donchian high AND above S3 (support hold) AND volume > 1.6x MA
            if close[i] > donch_high[i] and close[i] > s3_6h and volume[i] > 1.6 * volume_ma[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 6h Donchian low AND below R3 (resistance hold) AND volume > 1.6x MA
            elif close[i] < donch_low[i] and close[i] < r3_6h and volume[i] > 1.6 * volume_ma[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 6h Donchian low OR below S4 (S3 - 0.5*range)
            s4 = s3 - 0.5 * prev_range
            s4_6h = align_htf_to_ltf(prices, df_1w, np.full(len(df_1w), s4))[i]
            if close[i] < donch_low[i] or close[i] < s4_6h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 6h Donchian high OR above R4 (R3 + 0.5*range)
            r4 = r3 + 0.5 * prev_range
            r4_6h = align_htf_to_ltf(prices, df_1w, np.full(len(df_1w), r4))[i]
            if close[i] > donch_high[i] or close[i] > r4_6h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_S3R3_Donchian20_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0