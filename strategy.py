#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume spike confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > EMA50(1w) AND volume > 2.0x 20-period MA.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < EMA50(1w) AND volume > 2.0x 20-period MA.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 30-100 total trades over 4 years (7-25/year).
# Williams Alligator identifies trend alignment using smoothed medians, effective in both bull and bear markets.
# The 1w EMA50 ensures we only trade with the higher timeframe trend, avoiding counter-trend whipsaws.
# Volume confirmation validates breakout strength, reducing false signals.

name = "1d_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 1d: SMMA (Smoothed Moving Average) of median price
    # Jaw: SMMA of median, period=13, shift=8
    # Teeth: SMMA of median, period=8, shift=5
    # Lips: SMMA of median, period=5, shift=3
    median_price = (high + low) / 2
    
    def smma(arr, period):
        # Smoothed Moving Average: first value is SMA, then recursive smoothing
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # Initial SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply shifts: Jaw shifted by 8, Teeth by 5, Lips by 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Volume confirmation: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        vol_spike = volume_spike[i]
        
        # Alligator alignment
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        
        bullish_alignment = jaw_val < teeth_val and teeth_val < lips_val
        bearish_alignment = jaw_val > teeth_val and teeth_val > lips_val
        
        # Entry logic
        if position == 0:
            # Long: bullish alignment AND 1w uptrend AND volume spike
            if bullish_alignment and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND 1w downtrend AND volume spike
            elif bearish_alignment and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment OR 1w trend turns down
            if bearish_alignment or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR 1w trend turns up
            if bullish_alignment or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals