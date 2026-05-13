#!/usr/bin/env python3
"""
6h_1d_ElderRay_BullBearPower_1wTrend
Hypothesis: Elder Ray (Bull/Bear Power) on 6x with 1-day trend filter and 1-week regime filter.
In bull regimes (price > weekly EMA50), go long when Bull Power > 0 and Bear Power < 0.
In bear regimes (price < weekly EMA50), go short when Bear Power < 0 and Bull Power > 0.
This captures trend continuation in the dominant regime while avoiding counter-trend trades.
Target: 12-30 trades/year per symbol.
"""

name = "6h_1d_ElderRay_BullBearPower_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: 34 EMA
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema34_1d
    downtrend_1d = close_1d < ema34_1d
    
    # Get 1w data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w regime: 50 EMA
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    bull_regime = close_1w > ema50_1w
    bear_regime = close_1w < ema50_1w
    
    # Align 1d trend and 1w regime to 6h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    bull_regime_aligned = align_htf_to_ltf(prices, df_1w, bull_regime)
    bear_regime_aligned = align_htf_to_ltf(prices, df_1w, bear_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get aligned values
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        bull_reg = bull_regime_aligned[i]
        bear_reg = bear_regime_aligned[i]
        
        if position == 0:
            # LONG: bull regime + 1d uptrend + Bull Power > 0 and Bear Power < 0
            if bull_reg and uptrend and bull_power[i] > 0 and bear_power[i] < 0:
                signals[i] = 0.25
                position = 1
            # SHORT: bear regime + 1d downtrend + Bear Power < 0 and Bull Power > 0
            elif bear_reg and downtrend and bear_power[i] < 0 and bull_power[i] > 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: regime turns bearish or 1d trend turns down or Elder Ray breaks down
            if not bull_reg or not uptrend or bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: regime turns bullish or 1d trend turns up or Elder Ray breaks down
            if not bear_reg or not downtrend or bear_power[i] >= 0 or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals