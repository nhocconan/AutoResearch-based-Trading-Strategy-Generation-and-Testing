#!/usr/bin/env python3
name = "6h_Supertrend_DMI_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1D Trend: EMA34 > EMA89
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(df_1d['close']).ewm(span=89, adjust=False, min_periods=89).mean().values
    trend_1d = ema34_1d > ema89_1d
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 6H Supertrend (ATR=10, multiplier=3.0)
    atr_period = 10
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    hl2 = (high + low) / 2
    upper = hl2 + 3.0 * atr
    lower = hl2 - 3.0 * atr
    
    supertrend = np.zeros(n)
    dir = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    dir[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            dir[i] = 1
        else:
            dir[i] = -1
        
        if dir[i] == 1:
            supertrend[i] = max(lower[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
    
    # 6H DMI (ADX for trend strength, +DI/-DI for direction)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr_dmi = np.zeros(n)
    atr_dmi[:atr_period] = np.nan
    for i in range(atr_period, n):
        atr_dmi[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=atr_period, adjust=False).mean() / atr_dmi)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=atr_period, adjust=False).mean() / atr_dmi)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=atr_period, adjust=False).mean()
    
    # Volume filter: volume > 1.5x 50-period average
    vol_ma50 = np.zeros(n)
    for i in range(n):
        if i < 50:
            vol_ma50[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma50[i] = np.mean(volume[i-49:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(vol_ma50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Supertrend direction
        st_uptrend = close[i] > supertrend[i]
        st_downtrend = close[i] < supertrend[i]
        
        # DMI direction and strength
        dmi_bullish = plus_di[i] > minus_di[i]
        dmi_bearish = plus_di[i] < minus_di[i]
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: 1D uptrend + Supertrend uptrend + DMI bullish + Strong trend + Volume surge
            if (trend_1d_aligned[i] and st_uptrend and dmi_bullish and 
                strong_trend and volume[i] > 1.5 * vol_ma50[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1D downtrend + Supertrend downtrend + DMI bearish + Strong trend + Volume surge
            elif (not trend_1d_aligned[i] and st_downtrend and dmi_bearish and 
                  strong_trend and volume[i] > 1.5 * vol_ma50[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1D trend turns down OR Supertrend flips OR DMI turns bearish OR ADX weakens
            if (not trend_1d_aligned[i] or not st_uptrend or not dmi_bullish or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1D trend turns up OR Supertrend flips OR DMI turns bullish OR ADX weakens
            if (trend_1d_aligned[i] or not st_downtrend or not dmi_bearish or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals