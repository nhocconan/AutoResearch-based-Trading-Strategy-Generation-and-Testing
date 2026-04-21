#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Use Camarilla pivot levels from 1d timeframe (R1/S1) as entry triggers on 12h, with 1d EMA34 trend filter and volume confirmation. Designed to capture breakouts from key daily pivot levels in trending markets, with volume surge confirming institutional interest. Works in bull/bear markets by following higher timeframe trend (1d EMA) while using Camarilla levels for precise entry/exit. Target 15-30 trades/year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Calculate Camarilla pivot levels (R1, S1) from 1d OHLC ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculation
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume confirmation: 20-period volume average on 12h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_34_1d_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above R1 + volume spike > 1.5 + price above 1d EMA34
            if (price_close > r1_level and 
                vol_spike > 1.5 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume spike > 1.5 + price below 1d EMA34
            elif (price_close < s1_level and 
                  vol_spike > 1.5 and 
                  price_close < trend_1d):
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0