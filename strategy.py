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
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_14 = tr
    plus_di_14 = np.full(len(df_1w), np.nan)
    minus_di_14 = np.full(len(df_1w), np.nan)
    dx_14 = np.full(len(df_1w), np.nan)
    
    if len(df_1w) >= 14:
        # Smooth +DM, -DM, TR
        plus_dm_smooth = np.full(len(df_1w), np.nan)
        minus_dm_smooth = np.full(len(df_1w), np.nan)
        tr_smooth = np.full(len(df_1w), np.nan)
        
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
    
    adx_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 27:  # Need 14 + 14 for smoothing
        adx_1w[26] = np.mean(dx_14[14:28])
        for i in range(27, len(df_1w)):
            adx_1w[i] = (adx_1w[i-1] * 13 + dx_14[i]) / 14
    
    # Align weekly indicators to 6h timeframe
    atr_6h = align_htf_to_ltf(prices, df_1w, atr_1w)
    adx_6h = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 6-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 6-hour volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_6h[i]) or
            np.isnan(adx_6h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_6h[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 80% of 20-period MA)
        if volume[i] < 0.8 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Skip low trend strength (ADX < 25)
        if adx_6h[i] < 25:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above 6h Donchian high AND weekly ADX > 25
            if close[i] > donch_high[i] and adx_6h[i] > 25:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 6h Donchian low AND weekly ADX > 25
            elif close[i] < donch_low[i] and adx_6h[i] > 25:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 6h Donchian low OR weekly ADX < 20
            if close[i] < donch_low[i] or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 6h Donchian high OR weekly ADX < 20
            if close[i] > donch_high[i] or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Donchian_ADX_Trend_Filter"
timeframe = "6h"
leverage = 1.0