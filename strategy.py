#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Elder Ray Bull/Bear Power with 1d EMA34 trend filter.
# Long when Bull Power > 0 (close > EMA13) AND price > 1d EMA34 (uptrend).
# Short when Bear Power < 0 (close < EMA13) AND price < 1d EMA34 (downtrend).
# Exit when power reverses or price crosses 1d EMA34 opposite.
# Uses discrete position size 0.25. Elder Ray measures bull/bear strength via EMA13.
# 1d EMA34 filter ensures trading with higher timeframe trend to avoid whipsaws.
# 6h timeframe targets 12-37 trades/year to minimize fee drag.
# Works in bull markets (buy strength in uptrend) and bear markets (sell weakness in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data once before loop for EMA13 (Elder Ray)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Get 1d data once before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 12h Indicators: EMA13 for Elder Ray ===
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = low_12h - ema13_12h
    
    # === 1d Indicators: EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to primary timeframe (6h)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 34  # EMA34 needs 34 periods
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        ema34 = ema34_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Bear Power >= 0 (weakness) OR price crosses below 1d EMA34
            if (bear_power >= 0) or (price < ema34):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Bull Power <= 0 (strength) OR price crosses above 1d EMA34
            if (bull_power <= 0) or (price > ema34):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 (strength) AND price > 1d EMA34 (uptrend)
            if (bull_power > 0) and (price > ema34):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power < 0 (weakness) AND price < 1d EMA34 (downtrend)
            elif (bear_power < 0) and (price < ema34):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_12hElderRay_BullBearPower_1dEMA34_TrendFilter_V1"
timeframe = "6h"
leverage = 1.0