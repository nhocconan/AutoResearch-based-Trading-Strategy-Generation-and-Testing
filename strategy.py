#!/usr/bin/env python3
"""
6h_ElderRay_BullPower_BearPower_1dTrend
Hypothesis: Use Elder Ray index (Bull Power = High - EMA(13), Bear Power = Low - EMA(13)) with 1d EMA50 trend filter on 6h timeframe.
Long when Bull Power > 0 and Bear Power < 0 (strong bullish momentum) and price above 1d EMA50.
Short when Bear Power < 0 and Bull Power < 0 (strong bearish momentum) and price below 1d EMA50.
Exit when momentum weakens (Bear Power > 0 for longs, Bull Power < 0 for shorts).
Elder Ray captures institutional buying/selling pressure, and 1d EMA50 filters for higher-timeframe trend.
Designed to work in both bull and bear markets by following the dominant trend on 1d.
Target ~15-25 trades/year on 6h by requiring strong momentum alignment.
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
    
    # === Elder Ray calculation on 6h ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
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
        
        price_close = close[i]
        trend_1d = ema_50_1d_aligned[i]
        bp = bull_power[i]
        bearp = bear_power[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND Bear Power < 0 (no selling pressure) AND price above 1d EMA50
            if (bp > 0 and bearp < 0 and price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) AND Bull Power < 0 (no buying pressure) AND price below 1d EMA50
            elif (bearp < 0 and bp < 0 and price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when momentum weakens: for long, Bear Power becomes positive (selling pressure appears)
            # for short, Bull Power becomes positive (buying pressure appears)
            if position == 1 and bearp > 0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and bp > 0:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullPower_BearPower_1dTrend"
timeframe = "6h"
leverage = 1.0