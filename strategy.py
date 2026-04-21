#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA 34 for trend
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for pivot levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Camarilla pivot levels from previous day
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    # Resistance and support levels
    r1 = pivot + (high_1d - low_1d) * 1.1 / 12
    s1 = pivot - (high_1d - low_1d) * 1.1 / 12
    r2 = pivot + (high_1d - low_1d) * 1.1 / 6
    s2 = pivot - (high_1d - low_1d) * 1.1 / 6
    
    # Align pivot levels
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_1w_aligned[i]
        atr_val = atr_14_aligned[i]
        vol_ratio_val = vol_ratio[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        
        if position == 0:
            # Enter long: price touches S1/S2 with volume and weekly uptrend
            if (price_close <= s1_val * 1.005 and price_close >= s2_val * 0.995 and
                vol_ratio_val > 1.3 and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches R1/R2 with volume and weekly downtrend
            elif (price_close >= r1_val * 0.995 and price_close <= r2_val * 1.005 and
                  vol_ratio_val > 1.3 and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse touch or volatility expansion
            if position == 1 and (price_close >= r1_val * 0.995 or atr_val > 3 * atr_14_aligned[i-1] if i > 0 else False):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close <= s1_val * 1.005 or atr_val > 3 * atr_14_aligned[i-1] if i > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Touch_WeeklyEMA_Trend"
timeframe = "12h"
leverage = 1.0