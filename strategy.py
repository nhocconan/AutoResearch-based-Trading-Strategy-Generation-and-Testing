#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_v1
Hypothesis: On 1h timeframe, price breaking above Camarilla R1 or below S1 with 4h EMA50 trend filter and 1d chop regime filter (CHOP<38.2 = trending) captures institutional breakouts in both bull and bear markets. Uses discrete sizing (0.20) to minimize fee churn. Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Load HTF data ONCE before loop ===
    # 4h for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h EMA50 for trend filter ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d Chop regime filter (EHLERS) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true range over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Chop = 100 * log10(sum(tr14) / (atr14 * 14)) / log10(14)
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Camarilla levels on 1h (primary) ===
    high_prev = prices['high'].shift(1).values
    low_prev = prices['low'].shift(1).values
    close_prev = prices['close'].shift(1).values
    
    # Camarilla R1, S1
    R1 = close_prev + 1.1 * (high_prev - low_prev) / 12
    S1 = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(R1[i]) or np.isnan(S1[i]) or
            np.isnan(high_prev[i]) or np.isnan(low_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_open = prices['open'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        price_close = prices['close'].iloc[i]
        ema_50 = ema_50_4h_aligned[i]
        chop_val = chop_aligned[i]
        r1 = R1[i]
        s1 = S1[i]
        
        # Regime filter: only trade when chop < 38.2 (trending market)
        if chop_val >= 38.2:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above 4h EMA50
            if price_high > r1 and price_close > ema_50:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + below 4h EMA50
            elif price_low < s1 and price_close < ema_50:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit when price re-enters Camarilla levels or trend/chop changes
            if position == 1:
                if price_low < r1 or price_close < ema_50 or chop_val >= 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if price_high > s1 or price_close > ema_50 or chop_val >= 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_v1"
timeframe = "1h"
leverage = 1.0