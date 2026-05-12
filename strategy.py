#!/usr/bin/env python3
# 1d Williams Alligator + RSI + 1w Trend Filter
# Strategy uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and momentum,
# combined with RSI for overbought/oversold conditions and 1w trend filter for higher timeframe bias.
# Designed to work in both bull and bear markets by only taking trades aligned with the weekly trend.
# Entry: Long when Lips cross above Jaw/Teeth AND RSI < 50 (bullish momentum) AND price > weekly EMA50
#        Short when Lips cross below Jaw/Teeth AND RSI > 50 (bearish momentum) AND price < weekly EMA50
# Exit: When Lips cross back in opposite direction or RSI reaches extreme levels (30/70)
# Position sizing: Discrete levels (0.25) to minimize churn and manage risk

name = "1d_WilliamsAlligator_RSI_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator: SMAs of median price (HLC/3) with different periods
    median_price = (high + low + close) / 3.0
    
    # Jaw: Blue line - 13-period SMMA, shifted 8 bars ahead
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift 8 bars forward
    
    # Teeth: Red line - 8-period SMMA, shifted 5 bars ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift 5 bars forward
    
    # Lips: Green line - 5-period SMMA, shifted 3 bars ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift 3 bars forward
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need enough data for Alligator (13+8) and RSI
    
    for i in range(start_idx, n):
        # Skip if 1w trend data not ready
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(lips_vals[i]) or np.isnan(jaw_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish crossover: Lips cross above Teeth and Jaw
            lips_above_teeth = lips_vals[i] > teeth_vals[i]
            teeth_above_jaw = teeth_vals[i] > jaw_vals[i]
            lips_above_jaw_prev = lips_vals[i-1] <= jaw_vals[i-1]  # was below or equal
            
            bullish_cross = lips_above_teeth and teeth_above_jaw and lips_above_jaw_prev
            
            # Bearish crossover: Lips cross below Teeth and Jaw
            lips_below_teeth = lips_vals[i] < teeth_vals[i]
            teeth_below_jaw = teeth_vals[i] < jaw_vals[i]
            lips_below_jaw_prev = lips_vals[i-1] >= jaw_vals[i-1]  # was above or equal
            
            bearish_cross = lips_below_teeth and teeth_below_jaw and lips_below_jaw_prev
            
            if bullish_cross and rsi[i] < 50 and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif bearish_cross and rsi[i] > 50 and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Lips cross below Jaw OR RSI overbought (>70)
            lips_below_jaw = lips_vals[i] < jaw_vals[i]
            rsi_overbought = rsi[i] > 70
            
            if lips_below_jaw or rsi_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Lips cross above Jaw OR RSI oversold (<30)
            lips_above_jaw = lips_vals[i] > jaw_vals[i]
            rsi_oversold = rsi[i] < 30
            
            if lips_above_jaw or rsi_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals