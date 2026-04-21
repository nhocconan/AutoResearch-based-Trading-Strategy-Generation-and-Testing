#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dTrend_Volume_Confirmation
Hypothesis: Use 12h Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation. Designed to capture breakouts in strong trends while avoiding counter-trend trades. Works in bull markets by following uptrend breakouts and in bear markets by following downtrend breakdowns. Target 15-25 trades/year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Donchian(20) on 12h ===
    high = prices['high'].values
    low = prices['low'].values
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_50_1d_aligned[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume spike + price above 1d EMA50
            if (price_close > donchian_high and 
                vol_spike > 1.5 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume spike + price below 1d EMA50
            elif (price_close < donchian_low and 
                  vol_spike > 1.5 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses back through Donchian midpoint
            midpoint = (donchian_high + donchian_low) / 2
            if position == 1 and price_close < midpoint:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dTrend_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0