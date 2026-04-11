#!/usr/bin/env python3
# 1d_1w_alligator_volume_v1
# Strategy: 1d Williams Alligator with volume confirmation and weekly trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Williams Alligator (SMAs of median price) identifies trends via jaw/teeth/lips alignment. 
# In uptrend: Lips > Teeth > Jaw. In downtrend: Lips < Teeth < Jaw. 
# We enter long when bullish alignment + volume confirmation, short when bearish alignment + volume confirmation.
# Weekly trend filter ensures we only trade in direction of higher timeframe trend to avoid counter-trend trades.
# Low frequency (~10-25/year) to minimize fee drift.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_alligator_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate median price: (high + low) / 2
    median_price = (high + low) / 2
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator components on daily timeframe
    # Jaw: 13-period SMMA of median, shifted 8 bars
    # Teeth: 8-period SMMA of median, shifted 5 bars  
    # Lips: 5-period SMMA of median, shifted 3 bars
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    def smma(array, period):
        """Smoothed Moving Average"""
        result = np.full_like(array, np.nan, dtype=float)
        if len(array) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(array[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(array)):
            result[i] = (result[i-1] * (period-1) + array[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply shifts as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set shifted values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        uptrend_weekly = close[i] > ema_50_1w_aligned[i]
        downtrend_weekly = close[i] < ema_50_1w_aligned[i]
        
        # Alligator alignment signals
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Entry logic: Alligator alignment + volume + weekly trend alignment
        if bullish_alignment and vol_confirm[i] and uptrend_weekly and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_alignment and vol_confirm[i] and downtrend_weekly and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Alligator alignment breaks or volume weakens
        elif position == 1 and (not bullish_alignment or not vol_confirm[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_alignment or not vol_confirm[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals