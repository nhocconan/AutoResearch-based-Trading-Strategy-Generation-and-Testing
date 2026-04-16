#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull Power/Bear Power) with 1w EMA34 trend filter.
# Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1w EMA34
# Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND price < 1w EMA34
# Exit when momentum divergence occurs (Bull Power <= 0 for long, Bear Power <= 0 for short)
# Uses discrete position size 0.25. Elder Ray measures bull/bear power via EMA13.
# 1w timeframe filter ensures trading only with higher timeframe trend to avoid whipsaws.
# 6h timeframe targets 12-30 trades/year to minimize fee drag.
# Works in bull markets (catch uptrends) and bear markets (catch downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Elder Ray (EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data once before loop for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Elder Ray (Bull Power, Bear Power) ===
    # EMA13 of close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = EMA13 - Low
    bear_power_1d = ema13_1d - low_1d
    
    # === 1w Indicators: EMA34 for trend filter ===
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to primary timeframe (6h)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40  # EMA13 needs sufficient warmup + EMA34
    
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
            # Exit when bull power turns negative (loss of bullish momentum)
            if bull_power <= 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when bear power turns negative (loss of bearish momentum)
            if bear_power <= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull power > 0 AND Bear power < 0 (bullish momentum) AND price > 1w EMA34 (uptrend filter)
            if (bull_power > 0) and (bear_power < 0) and (price > ema34):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear power > 0 AND Bull power < 0 (bearish momentum) AND price < 1w EMA34 (downtrend filter)
            elif (bear_power > 0) and (bull_power < 0) and (price < ema34):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dElderRay_BullBearPower_1wEMA34_TrendFilter_V1"
timeframe = "6h"
leverage = 1.0