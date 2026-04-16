#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 1w trend filter.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (open < EMA13) AND price > 1w EMA34 (uptrend).
# Short when Bear Power < 0 (open < EMA13) AND Bull Power < 0 (close < EMA13) AND price < 1w EMA34 (downtrend).
# Uses discrete position size 0.25. Exits when Elder Power signals reverse or price crosses 1w EMA34.
# Elder Ray measures bull/bear strength relative to EMA13. 1w EMA34 filter ensures trading with higher timeframe trend.
# 6h timeframe targets 12-37 trades/year to minimize fee drag. Works in bull markets via longs, bear via shorts.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Elder Ray (EMA13) ===
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = close_1d - ema13_1d  # Bull Power = Close - EMA13
    bear_power_1d = open_1d - ema13_1d   # Bear Power = Open - EMA13
    
    # === 1w Indicators: EMA34 Trend Filter ===
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to primary timeframe (6h)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(13, 34)  # EMA13 needs 13, EMA34 needs 34
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        ema34_1w = ema34_1w_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Bull Power <= 0 OR Bear Power >= 0 OR price <= 1w EMA34 (trend change)
            if (bull_power <= 0) or (bear_power >= 0) or (price <= ema34_1w):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Bull Power >= 0 OR Bear Power <= 0 OR price >= 1w EMA34 (trend change)
            if (bull_power >= 0) or (bear_power <= 0) or (price >= ema34_1w):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 AND price > 1w EMA34 (uptrend)
            if (bull_power > 0) and (bear_power < 0) and (price > ema34_1w):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power < 0 AND Bull Power < 0 AND price < 1w EMA34 (downtrend)
            elif (bear_power < 0) and (bull_power < 0) and (price < ema34_1w):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dElderRay_BullBearPower_1wEMA34TrendFilter_V1"
timeframe = "6h"
leverage = 1.0