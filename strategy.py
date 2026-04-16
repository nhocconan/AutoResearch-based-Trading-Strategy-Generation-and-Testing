#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter.
# Long when Lips > Teeth > Jaw (bullish alignment) AND close > EMA50 (uptrend).
# Short when Lips < Teeth < Jaw (bearish alignment) AND close < EMA50 (downtrend).
# Exit when Alligator alignment breaks or price crosses EMA50.
# Uses discrete position size 0.25. Alligator identifies trend phases; EMA50 filters counter-trend noise.
# 1d timeframe targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull markets (capture uptrends) and bear markets (capture downtrends) by trading with Alligator alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data once before loop for Williams Alligator (SMMA13, SMMA8, SMMA5)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 1d data once before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1w Indicators: Williams Alligator (using SMMA) ===
    # Jaw: SMMA of close, period 13
    # Teeth: SMMA of close, period 8
    # Lips: SMMA of close, period 5
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan, dtype=float)
        result = np.full_like(values, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (Prev_SMMA*(period-1) + Current_Close) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw_1w = smma(close_1w, 13)
    teeth_1w = smma(close_1w, 8)
    lips_1w = smma(close_1w, 5)
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        ema50 = ema50_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Alligator alignment breaks bullish (Lips <= Teeth or Teeth <= Jaw) OR price < EMA50 (trend break)
            if (lips <= teeth) or (teeth <= jaw) or (price < ema50):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Alligator alignment breaks bearish (Lips >= Teeth or Teeth >= Jaw) OR price > EMA50 (trend break)
            if (lips >= teeth) or (teeth >= jaw) or (price > ema50):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bullish alignment (Lips > Teeth > Jaw) AND price > EMA50 (uptrend)
            if (lips > teeth) and (teeth > jaw) and (price > ema50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bearish alignment (Lips < Teeth < Jaw) AND price < EMA50 (downtrend)
            elif (lips < teeth) and (teeth < jaw) and (price < ema50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_1wWilliamsAlligator_1dEMA50_TrendFilter_V1"
timeframe = "1d"
leverage = 1.0