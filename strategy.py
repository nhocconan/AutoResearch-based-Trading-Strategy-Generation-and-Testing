#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) AND 1d close > 1d EMA50 AND volume > 1.5x 20 EMA
# Short when Jaw > Teeth > Lips (bearish alignment) AND 1d close < 1d EMA50 AND volume > 1.5x 20 EMA
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year per symbol.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Williams Alligator catches trends early with smoothed medians, reducing whipsaw.
# 1d EMA50 filter ensures we only trade with the higher timeframe trend.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get daily close array for trend filter
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 12h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods
    # Lips: 5-period SMMA smoothed by 3 periods
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan, dtype=float)
        result = np.full_like(values, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current value) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Calculate Alligator components
    jaw = smma(close, 13)  # 13-period SMMA
    jaw = smma(jaw, 8)     # smoothed by 8 periods
    
    teeth = smma(close, 8)  # 8-period SMMA
    teeth = smma(teeth, 5)  # smoothed by 5 periods
    
    lips = smma(close, 5)   # 5-period SMMA
    lips = smma(lips, 3)    # smoothed by 3 periods
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Jaw < Teeth < Lips (bullish alignment) AND 1d uptrend AND volume spike
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Jaw > Teeth > Lips (bearish alignment) AND 1d downtrend AND volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment OR 1d trend changes to downtrend
            if (jaw[i] > teeth[i] or teeth[i] > lips[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment OR 1d trend changes to uptrend
            if (jaw[i] < teeth[i] or teeth[i] < lips[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals