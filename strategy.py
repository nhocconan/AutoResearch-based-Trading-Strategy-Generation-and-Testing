#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Bull/Bear Power with 1d EMA Trend Filter
# Hypothesis: Elder Ray's Bull/Bear Power measures trend strength by comparing price to EMA.
# Combined with 1d EMA trend filter to trade in direction of higher timeframe trend.
# Works in both bull and bear markets by following the dominant trend.
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag.

name = "6h_elder_ray_1d_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA(20) on 1d close
    close_1d = df_1d['close'].values
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    
    # Align 1d EMA to 6h
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Calculate EMA(13) for 6h Bull/Bear Power
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_20_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive (bullish momentum fading) or trend changes
            if bear_power[i] > 0 or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: Bull Power turns negative (bearish momentum fading) or trend changes
            if bull_power[i] < 0 or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Enter long in uptrend when Bull Power is strong (buying pressure)
            if close[i] > ema_20_aligned[i]:  # Uptrend
                if bull_power[i] > 0 and bear_power[i] < 0:  # Strong bullish momentum
                    position = 1
                    signals[i] = 0.25
            # Enter short in downtrend when Bear Power is strong (selling pressure)
            elif close[i] < ema_20_aligned[i]:  # Downtrend
                if bear_power[i] > 0 and bull_power[i] < 0:  # Strong bearish momentum
                    position = -1
                    signals[i] = -0.25
    
    return signals