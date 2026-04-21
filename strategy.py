#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA20_Trend_Volume
Hypothesis: Use Camarilla pivot levels (R1/S1) from 12h timeframe as entry triggers on 4h, with 12h EMA20 trend filter and volume confirmation. Designed to capture breakouts from key 12h pivot levels in trending markets, with volume surge confirming institutional interest. Target 18-45 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h trend filter: 20-period EMA ===
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # === Calculate Camarilla pivot levels (R1, S1) from 12h OHLC ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point calculation
    pp = (high_12h + low_12h + close_12h) / 3.0
    r1 = close_12h + (high_12h - low_12h) * 1.1 / 12
    s1 = close_12h - (high_12h - low_12h) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # === Volume confirmation: 20-period volume average on 4h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if indicators not ready
        if (np.isnan(ema_20_12h_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_12h = ema_20_12h_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above R1 + volume spike > 1.5 + price above 12h EMA20
            if (price_close > r1_level and 
                vol_spike > 1.5 and 
                price_close > trend_12h):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume spike > 1.5 + price below 12h EMA20
            elif (price_close < s1_level and 
                  vol_spike > 1.5 and 
                  price_close < trend_12h):
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

name = "4h_Camarilla_R1_S1_Breakout_12hEMA20_Trend_Volume"
timeframe = "4h"
leverage = 1.0