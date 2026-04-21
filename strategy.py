#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Pullback_1dTrend_Volume
Hypothesis: Buy pullbacks to S1/S2 in uptrend, sell rallies to R1/R2 in downtrend using 1d EMA50 trend filter. Volume confirmation requires 2x average volume. Target 25-40 trades/year on 4h for controlled frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Calculate Camarilla pivot levels from 1d OHLC ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculation
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    r2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    s2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # === Volume confirmation: 20-period volume average on 4h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after EMA warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_50_1d_aligned[i]
        r1_level = r1_aligned[i]
        r2_level = r2_aligned[i]
        s1_level = s1_aligned[i]
        s2_level = s2_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Pullback to S1/S2 in uptrend with volume spike
            if (price_close >= s1_level and price_close <= s2_level and
                price_close > trend_1d and
                vol_spike >= 2.0):
                signals[i] = 0.25
                position = 1
            # Short: Rally to R1/R2 in downtrend with volume spike
            elif (price_close <= r1_level and price_close >= r2_level and
                  price_close < trend_1d and
                  vol_spike >= 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to pivot point (PP)
            pp_level = pp_aligned[i]
            if position == 1 and price_close < pp_level:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > pp_level:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Pullback_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0