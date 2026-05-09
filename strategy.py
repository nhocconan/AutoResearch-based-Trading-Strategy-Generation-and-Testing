#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with Daily Trend Filter and Volume Spike
# Uses 12h Williams Alligator (Jaw/Teeth/Lips) for trend identification,
# daily EMA34 for trend alignment, and volume spike for confirmation.
# Designed for 12-37 trades/year to avoid fee drag.
# Works in bull markets (Alligator mouth opens up) and bear markets (mouth opens down).
name = "12h_WilliamsAlligator_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(df_daily['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Williams Alligator on 12h data
    # Jaw = 13-period SMMA (smoothed moving average) of median price, shifted 8 bars
    # Teeth = 8-period SMMA of median price, shifted 5 bars
    # Lips = 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts (Jaw: 8, Teeth: 5, Lips: 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align daily EMA to 12h
    # Note: Williams Alligator is already calculated on 12h data, no alignment needed
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13+8, 8+5, 5+3)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_12h[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (Alligator mouth opens up) with daily uptrend and volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema34_12h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (Alligator mouth opens down) with daily downtrend and volume spike
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and close[i] < ema34_12h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator mouth closes (Lips < Teeth) OR daily trend turns down
            if lips[i] < teeth[i] or close[i] < ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator mouth closes (Jaw < Teeth) OR daily trend turns up
            if jaw[i] < teeth[i] or close[i] > ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals