#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Bull/Bear Power with 1d trend filter
# Uses Williams Alligator (jaw/teeth/lips) to identify trend absence/presence
# Elder Ray measures bull/bear power relative to 13-period EMA
# 1d EMA50 provides higher-timeframe trend filter to avoid counter-trend trades
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 total) with discrete sizing (0.25)
# Works in bull markets by buying when bull power > 0 and Alligator aligned bullish
# Works in bear markets by selling when bear power > 0 and Alligator aligned bearish
# The 1d EMA50 filter ensures we only trade in the direction of the higher-timeframe trend

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs of median price (hl2) with different periods
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    hl2 = (high + low) / 2
    
    # Smoothed Moving Average (SMMA) - same as RMA/Wilder's MA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(hl2, 13)
    teeth = smma(hl2, 8)
    lips = smma(hl2, 5)
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Alligator alignment
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Jaw > Teeth > Lips
        bullish_alligator = (lips_aligned[i] > teeth_aligned[i] and 
                            teeth_aligned[i] > jaw_aligned[i])
        bearish_alligator = (jaw_aligned[i] > teeth_aligned[i] and 
                            teeth_aligned[i] > lips_aligned[i])
        
        if position == 0:
            # Long: Bull power > 0 + Bullish Alligator + price above 1d EMA50 (uptrend)
            if (bull_power[i] > 0 and bullish_alligator and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear power > 0 + Bearish Alligator + price below 1d EMA50 (downtrend)
            elif (bear_power[i] > 0 and bearish_alligator and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull power <= 0 OR Alligator loses bullish alignment OR price below 1d EMA50
            if (bull_power[i] <= 0 or not bullish_alligator or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear power <= 0 OR Alligator loses bearish alignment OR price above 1d EMA50
            if (bear_power[i] <= 0 or not bearish_alligator or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals