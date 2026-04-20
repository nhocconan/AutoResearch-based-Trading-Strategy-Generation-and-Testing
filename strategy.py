#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d trend filter
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low (13-period EMA on 1d)
# - Long when Bull Power > 0 and Bear Power < 0 (both bullish) and 1d close > 1d EMA50
# - Short when Bear Power > 0 and Bull Power < 0 (both bearish) and 1d close < 1d EMA50
# - Exit when power signals reverse or 1d trend changes
# - Uses 1d for trend and power calculation, 6h for execution
# - Target: 20-40 trades per year per symbol (80-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Elder Ray Power calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 and EMA50 on 1d data
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Elder Ray Power
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or np.isnan(ema50_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema50_6h[i]
        bull_power = bull_power_6h[i]
        bear_power = bear_power_6h[i]
        
        if position == 0:
            # Long entry: Bull Power > 0, Bear Power < 0, and price above EMA50
            if bull_power > 0 and bear_power < 0 and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0, Bull Power < 0, and price below EMA50
            elif bear_power > 0 and bull_power < 0 and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Power signals reverse or trend changes
            if bull_power <= 0 or bear_power >= 0 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Power signals reverse or trend changes
            if bear_power <= 0 or bull_power >= 0 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dTrendFilter"
timeframe = "6h"
leverage = 1.0