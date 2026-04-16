#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams Alligator (Jaw/Teeth/Lips) with price cross confirmation.
# Long when price > Lips AND Lips > Teeth > Jaw (bullish alignment).
# Short when price < Lips AND Lips < Teeth < Jaw (bearish alignment).
# Exit when price crosses back below/above Lips or alignment breaks.
# Uses discrete position size 0.25. Williams Alligator identifies trending vs ranging markets.
# 1w timeframe filter ensures trading only with higher timeframe trend to avoid whipsaws.
# 1d timeframe targets 15-25 trades/year to minimize fee drag.
# Works in bull markets (catch uptrends) and bear markets (catch downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Williams Alligator (SMMA13, SMMA8, SMMA5)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Williams Alligator (Smoothed Moving Average) ===
    # Jaw: SMMA13 (13-period)
    # Teeth: SMMA8 (8-period) 
    # Lips: SMMA5 (5-period)
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw_1d = smma(close_1d, 13)
    teeth_1d = smma(close_1d, 8)
    lips_1d = smma(close_1d, 5)
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # SMMA13 needs sufficient warmup + EMA50
    
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
            # Exit when price crosses below Lips OR Alligator alignment breaks (Lips <= Teeth)
            if (price < lips) or (lips <= teeth):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price crosses above Lips OR Alligator alignment breaks (Lips >= Teeth)
            if (price > lips) or (lips >= teeth):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw AND price > Lips
            bullish_alignment = (lips > teeth) and (teeth > jaw)
            # Bearish alignment: Lips < Teeth < Jaw AND price < Lips
            bearish_alignment = (lips < teeth) and (teeth < jaw)
            
            # LONG: Bullish alignment AND price > Lips AND price > 1w EMA50 (uptrend filter)
            if bullish_alignment and (price > lips) and (price > ema50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bearish alignment AND price < Lips AND price < 1w EMA50 (downtrend filter)
            elif bearish_alignment and (price < lips) and (price < ema50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_1wWilliamsAlligator_JawTeethLips_1wEMA50_TrendFilter_V1"
timeframe = "1d"
leverage = 1.0