#!/usr/bin/env python3
# 6h_Alligator_RelativeStrength_1dTrend_Filter
# Hypothesis: Uses Williams Alligator (13,8,5 SMAs) to identify trend direction and strength,
# combined with 1d EMA50 trend filter and 6h relative strength (RSI vs 50) to avoid whipsaws.
# In bull markets: Alligator bullish alignment + price above 1d EMA50 + RSI > 50 = long.
# In bear markets: Alligator bearish alignment + price below 1d EMA50 + RSI < 50 = short.
# The Alligator's jaw-teeth-lips convergence filters sideways markets, reducing false signals.
# Target: 20-40 trades/year to minimize fee drag while maintaining edge in both bull/bear regimes.

name = "6h_Alligator_RelativeStrength_1dTrend_Filter"
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 6h data
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Alligator alignment signals
    # Bullish: Lips > Teeth > Jaw (all aligned upward)
    # Bearish: Lips < Teeth < Jaw (all aligned downward)
    alligator_bullish = (lips > teeth) & (teeth > jaw)
    alligator_bearish = (lips < teeth) & (teeth < jaw)
    
    # Relative strength filter: 6h RSI(14) vs 50
    def rsi(arr, period=14):
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(arr, np.nan)
        avg_loss = np.full_like(arr, np.nan)
        if len(arr) < period:
            return avg_gain / (avg_loss + 1e-10)
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        for i in range(period, len(arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    rsi_6h = rsi(close, 14)
    rsi_bullish = rsi_6h > 50
    rsi_bearish = rsi_6h < 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(rsi_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator bullish + price above 1d EMA50 + RSI > 50
            if alligator_bullish[i] and close[i] > ema_50_1d_aligned[i] and rsi_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + price below 1d EMA50 + RSI < 50
            elif alligator_bearish[i] and close[i] < ema_50_1d_aligned[i] and rsi_bearish[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator alignment breaks or price crosses 1d EMA50
            if not alligator_bullish[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator alignment breaks or price crosses 1d EMA50
            if not alligator_bearish[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals