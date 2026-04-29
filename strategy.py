#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation (>1.5x 20-period average)
# Williams Alligator uses smoothed medians (Jaw/Teeth/Lips) to identify trends: Lips > Teeth > Jaw = uptrend, reverse = downtrend
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume confirmation filters weak breakouts, reducing trade frequency
# Target: 75-150 total trades over 4 years (19-38/year) on 4h timeframe
# Works in both bull/bear: Alligator identifies trend direction, EMA filter ensures higher-timeframe alignment

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 4h timeframe
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Alligator lines: Jaw (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # Calculate 20-period average volume for confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 20)  # 1d EMA50, Alligator jaw warmup, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Alligator trend conditions
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Lips < Teeth < Jaw
        is_uptrend = curr_lips > curr_teeth > curr_jaw
        is_downtrend = curr_lips < curr_teeth < curr_jaw
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: trend changes to downtrend OR price closes below 1d EMA50
            if not is_uptrend or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend changes to uptrend OR price closes above 1d EMA50
            if not is_downtrend or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: uptrend + price above 1d EMA50 + volume confirmation
            if is_uptrend and curr_close > curr_ema_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price below 1d EMA50 + volume confirmation
            elif is_downtrend and curr_close < curr_ema_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals