#!/usr/bin/env python3
"""
4h_Pivot_R1_Breakout_Volume_Trend_1d
Hypothesis: Buy when price breaks above daily pivot R1 with volume spike and 1d EMA50 uptrend; sell when breaks below S1 with volume spike and 1d EMA50 downtrend. Uses Camarilla pivots from prior day for objective levels, volume confirmation to avoid fakeouts, and trend filter to align with higher timeframe direction. Designed to work in both bull and bear markets by following 1d trend while using intraday breakouts for entry. Target 20-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d OHLC data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Calculate Camarilla pivot levels from prior day ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Camarilla calculations
    range_ = high_1d - low_1d
    r1 = close_1d + (range_ * 1.1 / 12)
    s1 = close_1d - (range_ * 1.1 / 12)
    
    # Align pivot levels to 4h timeframe (use prior day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_50_1d_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Break above R1 + volume spike > 1.5 + price above 1d EMA50 (uptrend)
            if (price_close > r1_level and 
                vol_spike > 1.5 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 + volume spike > 1.5 + price below 1d EMA50 (downtrend)
            elif (price_close < s1_level and 
                  vol_spike > 1.5 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to pivot level (mean reversion)
            if position == 1 and price_close < r1_level:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > s1_level:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Pivot_R1_Breakout_Volume_Trend_1d"
timeframe = "4h"
leverage = 1.0