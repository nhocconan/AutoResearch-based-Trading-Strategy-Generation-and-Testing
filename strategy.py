#!/usr/bin/env python3
"""
6h_ElderRay_BullPower_BearPower_1dTrend
Hypothesis: Use Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h with 1d EMA50 trend filter.
Long when Bull Power > 0 and Bear Power < 0 and price > 1d EMA50 (bullish regime).
Short when Bear Power < 0 and Bull Power < 0 and price < 1d EMA50 (bearish regime).
Exit when power signals weaken or trend changes. Designed to capture institutional buying/selling pressure.
Target ~15-25 trades/year per symbol by requiring confluence of price action and trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
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
        br = bear_power[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND Bear Power < 0 (weak selling) AND above 1d EMA50
            if (bp > 0 and br < 0 and price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) AND Bull Power < 0 (weak buying) AND below 1d EMA50
            elif (br < 0 and bp < 0 and price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: weakening pressure or trend change
            if position == 1:
                # Exit long when buying pressure weakens (Bull Power <= 0) or trend turns bearish
                if bp <= 0 or price_close < trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25  # hold long
            else:  # position == -1
                # Exit short when selling pressure weakens (Bear Power >= 0) or trend turns bullish
                if br >= 0 or price_close > trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25  # hold short
    
    return signals

name = "6h_ElderRay_BullPower_BearPower_1dTrend"
timeframe = "6h"
leverage = 1.0