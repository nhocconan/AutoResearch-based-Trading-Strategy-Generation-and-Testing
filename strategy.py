#!/usr/bin/env python3
"""
6h_ElderRay_BullPower_BearPower_1dTrend
Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6m timeframe with 1d EMA trend filter.
Long when Bull Power > 0 and rising, Bear Power < 0 and falling, price above 1d EMA50.
Short when Bear Power < 0 and falling, Bull Power < 0 and rising, price below 1d EMA50.
Uses EMA13 for power calculation and EMA50 for trend filter. Designed to capture momentum in trending markets while avoiding counter-trend trades.
Works in bull/bear markets by following higher timeframe trend (1d EMA) while using Elder Ray for precise entry/exit.
Target 12-37 trades/year on 6h.
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
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Elder Ray Index: Bull Power and Bear Power (EMA13) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA13 for power calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_50_1d_aligned[i]
        bp = bull_power[i]
        bep = bear_power[i]
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0 and falling, price above 1d EMA50
            if (bp > 0 and 
                i > 50 and bp > bull_power[i-1] and  # Bull Power rising
                bep < 0 and 
                i > 50 and bep < bear_power[i-1] and  # Bear Power falling
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, Bull Power < 0 and rising, price below 1d EMA50
            elif (bep < 0 and 
                  i > 50 and bep < bear_power[i-1] and  # Bear Power falling
                  bp < 0 and 
                  i > 50 and bp > bull_power[i-1] and  # Bull Power rising
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when power signals diverge or trend changes
            if position == 1:
                # Exit long when Bull Power turns negative or Bear Power turns positive
                if bp <= 0 or bep >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when Bear Power turns positive or Bull Power turns negative
                if bep >= 0 or bp <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullPower_BearPower_1dTrend"
timeframe = "6h"
leverage = 1.0