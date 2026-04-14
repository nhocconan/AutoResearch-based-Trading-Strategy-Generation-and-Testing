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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR (14-period)
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily ADX (14-period)
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up = high_1d[i] - high_1d[i-1]
        down = low_1d[i-1] - low_1d[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
    
    tr_14 = np.zeros(len(df_1d))
    tr_14[0] = tr[0]
    for i in range(1, len(df_1d)):
        tr_14[i] = tr[i]
    
    atr_14 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_14[13] = np.mean(tr_14[:14])
        for i in range(14, len(df_1d)):
            atr_14[i] = (atr_14[i-1] * 13 + tr_14[i]) / 14
    
    plus_di_14 = np.full(len(df_1d), np.nan)
    minus_di_14 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        for i in range(13, len(df_1d)):
            if atr_14[i] > 0:
                plus_di_14[i] = 100 * (np.mean(plus_dm[i-13:i+1]) / atr_14[i])
                minus_di_14[i] = 100 * (np.mean(minus_dm[i-13:i+1]) / atr_14[i])
    
    dx_14 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        for i in range(13, len(df_1d)):
            if plus_di_14[i] + minus_di_14[i] > 0:
                dx_14[i] = 100 * abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])
    
    adx_14 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 27:
        adx_14[26] = np.mean(dx_14[13:27])
        for i in range(27, len(df_1d)):
            adx_14[i] = (adx_14[i-1] * 13 + dx_14[i]) / 14
    
    adx_12h = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate daily volume ratio (current volume / 20-day average)
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_ratio = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if not np.isnan(vol_ma_20[i]) and vol_ma_20[i] > 0:
            vol_ratio[i] = volume_1d[i] / vol_ma_20[i]
    
    vol_ratio_12h = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Calculate 4-hour Donchian channels (20-period) for entry timing
    donch_high_4h = np.full(n, np.nan)
    donch_low_4h = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high_4h[i] = np.max(high[i-19:i+1])
            donch_low_4h[i] = np.min(low[i-19:i+1])
    
    # Calculate 4-hour ATR (14-period) for stop loss
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr_4h = np.full(n, np.nan)
    if n >= 14:
        atr_4h[13] = np.mean(tr_4h[:14])
        for i in range(14, n):
            atr_4h[i] = (atr_4h[i-1] * 13 + tr_4h[i]) / 14
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_12h[i]) or
            np.isnan(adx_12h[i]) or
            np.isnan(vol_ratio_12h[i]) or
            np.isnan(donch_high_4h[i]) or
            np.isnan(donch_low_4h[i]) or
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 1.5% of price)
        if atr_12h[i] / close[i] < 0.015:
            signals[i] = 0.0
            continue
        
        # Skip weak trend (ADX < 25)
        if adx_12h[i] < 25:
            signals[i] = 0.0
            continue
        
        # Skip low volume (volume ratio < 1.5)
        if vol_ratio_12h[i] < 1.5:
            signals[i] = 0.0
            continue
        
        # Calculate daily pivot levels based on previous day's range
        prev_high = high_1d[i-1] if i > 0 else high_1d[0]
        prev_low = low_1d[i-1] if i > 0 else low_1d[0]
        prev_close = close_1d[i-1] if i > 0 else close_1d[0]
        prev_range = prev_high - prev_low
        
        # Camarilla-style pivot levels (R3/S3)
        r3 = prev_close + (prev_range * 1.1 / 4)
        s3 = prev_close - (prev_range * 1.1 / 4)
        
        # Align to 12h timeframe
        r3_12h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r3))[i]
        s3_12h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s3))[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high AND above S3 AND volume confirmation
            if close[i] > donch_high_4h[i] and close[i] > s3_12h and vol_ratio_12h[i] > 1.5:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 4h Donchian low AND below R3 AND volume confirmation
            elif close[i] < donch_low_4h[i] and close[i] < r3_12h and vol_ratio_12h[i] > 1.5:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 4h Donchian low OR below S3 OR ATR-based stop loss
            if close[i] < donch_low_4h[i] or close[i] < s3_12h or close[i] < (signals[i-1] * close[i-1] - 2.0 * atr_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 4h Donchian high OR above R3 OR ATR-based stop loss
            if close[i] > donch_high_4h[i] or close[i] > r3_12h or close[i] > (signals[i-1] * close[i-1] + 2.0 * atr_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_R3S3_Breakout_4hDonchian_VolumeFilter"
timeframe = "12h"
leverage = 1.0