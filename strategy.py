#!/usr/bin/env python3
# 1h_Choppy_Keltner_MeanReversion
# Hypothesis: In choppy markets (low ADX, high Choppiness Index), price reverts to the mean within Keltner Channels.
# Uses 4h and 1d for regime detection (trend vs chop) and 1h for precise mean-reversion entries.
# Works in bull via mean reversion in uptrend pullbacks, and in bear via bounces off lower bands in downtrend rallies.
# Avoids trend-following whipsaws by only trading in chop regimes, reducing false signals and fatigue.

name = "1h_Choppy_Keltner_MeanReversion"
timeframe = "1h"
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
    
    # === 4h Data for Regime Detection (Trend vs Chop) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # ADX(14) on 4h for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = minus_dm[i] = 0
            elif plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            else:
                plus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
        adx[:period] = np.nan
        return adx.values
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Choppiness Index (14) on 4h
    def calculate_choppiness(high, low, close, period=14):
        atr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if i < period:
                atr[i] = np.nan
            else:
                if i == period:
                    atr[i] = np.mean(tr[1:period+1])
                else:
                    atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        sum_atr = np.nansum(np.column_stack([atr[i-period+1:i+1] for i in range(period-1, len(high))]), axis=1)
        max_h = np.zeros_like(high)
        min_l = np.zeros_like(high)
        for i in range(len(high)):
            if i < period:
                max_h[i] = np.nan
                min_l[i] = np.nan
            else:
                max_h[i] = np.max(high[i-period+1:i+1])
                min_l[i] = np.min(low[i-period+1:i+1])
        
        chop = 100 * np.log10(sum_atr / (max_h - min_l)) / np.log10(period)
        return chop
    
    chop_4h = calculate_choppiness(high_4h, low_4h, close_4h, 14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    # === 1d Data for Trend Filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1h Keltner Channel (20, 2.0) ===
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = np.zeros_like(high)
    for i in range(1, len(high)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if i < 1:
            atr[i] = np.nan
        else:
            if i < 20:
                atr[i] = np.mean(tr[1:i+1]) if i >= 1 else np.nan
            else:
                atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    upper_keltner = ema_20 + 2.0 * atr
    lower_keltner = ema_20 - 2.0 * atr
    
    # === Session Filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(chop_4h_aligned[i]) or 
            np.isnan(ema_50_1h[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Regime filter: Chop when ADX < 20 and Choppiness > 50
        is_chop = (adx_4h_aligned[i] < 20) and (chop_4h_aligned[i] > 50)
        
        if position == 0 and in_session and is_chop:
            # LONG: Price at or below lower Keltner + above 1d EMA50 (buy dip in uptrend)
            if close[i] <= lower_keltner[i] and close[i] > ema_50_1h[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price at or above upper Keltner + below 1d EMA50 (sell rally in downtrend)
            elif close[i] >= upper_keltner[i] and close[i] < ema_50_1h[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Price touches upper Keltner or trend shifts (below EMA50)
            if close[i] >= upper_keltner[i] or close[i] < ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price touches lower Keltner or trend shifts (above EMA50)
            if close[i] <= lower_keltner[i] or close[i] > ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals