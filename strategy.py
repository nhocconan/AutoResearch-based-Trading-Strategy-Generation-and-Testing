#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Williams Alligator uses smoothed medians (Jaw=13, Teeth=8, Lips=5) to identify trends.
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment).
# Uses 1w EMA50 as trend filter to avoid counter-trend trades. Volume confirmation reduces false signals.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (~12-37/year).
# Works in both bull and bear markets by following the trend defined by Alligator alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator components (using SMMA = smoothed moving average)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    # Smoothed Moving Average (SMMA) - similar to RMA/Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            else:
                result[i] = np.nan
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply Alligator shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set first values to NaN where shift creates invalid data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(ema_50_1w_aligned[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish = lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish = lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]
        
        # Trend filter: price above/below 1w EMA50
        price_above_trend = close[i] > ema_50_1w_aligned[i]
        price_below_trend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x median of last 20 periods
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_confirm = volume[i] > 1.5 * vol_median
        
        # Long entry: bullish alignment + price above trend + volume confirmation
        if bullish and price_above_trend and volume_confirm and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish alignment + price below trend + volume confirmation
        elif bearish and price_below_trend and volume_confirm and position >= 0:
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite alignment or loss of trend
        elif position == 1 and (bearish or not price_above_trend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish or not price_below_trend):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0