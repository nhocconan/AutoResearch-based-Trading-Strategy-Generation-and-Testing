#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams Alligator + Elder Ray Power with 1d trend filter
    # Works in both bull and bear markets: Alligator identifies trend state,
    # Elder Ray measures bull/bear power, 1d EMA50 filters higher timeframe trend.
    # Low trade frequency (~20-40/year) avoids fee drag.
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA50 trend filter
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Williams Alligator (6h): Jaw(13), Teeth(8), Lips(5) - smoothed with SMMA
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is SMA
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Elder Ray Power (6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator sleeping: jaws, teeth, lips intertwined (no strong trend)
            # Alligator awakening: lines separate in specific order
            # Bullish: Lips > Teeth > Jaw (green > red > blue)
            # Bearish: Jaw > Teeth > Lips (blue > red > green)
            
            # Long: Bullish alignment + Bull Power > 0 + 1d Uptrend
            if (lips[i] > teeth[i] > jaw[i] and 
                bull_power[i] > 0 and 
                close[i] > ema_1d_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + Bear Power < 0 + 1d Downtrend
            elif (jaw[i] > teeth[i] > lips[i] and 
                  bear_power[i] < 0 and 
                  close[i] < ema_1d_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator returns to sleep (lines re-intertwine) OR Elder Power diverges
            if position == 1:
                # Exit long: Bearish power OR Alligator turns bearish
                if (bear_power[i] < 0 or 
                    jaw[i] > teeth[i] > lips[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Bullish power OR Alligator turns bullish
                if (bull_power[i] > 0 or 
                    lips[i] > teeth[i] > jaw[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_ElderRay_Power_1dEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0