#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trend direction.
# 1w EMA50 ensures alignment with higher timeframe trend.
# Volume spike >1.8 filters false signals. Discrete sizing 0.25.
# Designed to work in both bull (trend following) and bear (mean reversion at extremes).
name = "12h_Williams_Alligator_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Alligator parameters
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate Alligator lines using SMMA (Smoothed Moving Average)
    # SMMA is similar to Wilder's smoothing (alpha = 1/period)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate median price (typical price)
    median_price = (high + low) / 2.0
    
    # Calculate Alligator lines
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    # Apply shifts (Jaw: 8 bars, Teeth: 5 bars, Lips: 3 bars)
    jaw_shifted = np.roll(jaw, jaw_shift)
    teeth_shifted = np.roll(teeth, teeth_shift)
    lips_shifted = np.roll(lips, lips_shift)
    
    # Set NaN for shifted positions
    jaw_shifted[:jaw_shift] = np.nan
    teeth_shifted[:teeth_shift] = np.nan
    lips_shifted[:lips_shift] = np.nan
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, jaw_shift, teeth_shift, lips_shift, 50)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Lips > Teeth > Jaw (bullish alignment) with trend alignment and volume spike
            if (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and 
                close[i] > ema50_1w_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (bearish alignment) with trend alignment and volume spike
            elif (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment (Lips < Teeth < Jaw) OR trend fails
            if lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment (Lips > Teeth > Jaw) OR trend fails
            if lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals