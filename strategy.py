#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter (EMA50) and volume confirmation
# Uses Alligator (Jaw/Teeth/Lips) to identify trend absence/presence - trades only when all three aligned
# 1d EMA50 ensures trades align with daily trend to avoid counter-trend whipsaws
# Volume spike (1.8x 20-period avg) confirms institutional participation
# Discrete sizing 0.25 balances profit potential with fee drag (target 50-150 trades/4 years)
# Works in bull/bear by only taking entries in direction of 1d trend when Alligator is "awake"

name = "12h_Williams_Alligator_1dEMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_series = pd.Series(df_1d['close'])
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (Jaw=13, Teeth=8, Lips=5)
    # Alligator = Smoothed Moving Average (SMMA) with specific periods
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Jaw (Blue) - 13-period SMMA of median price
    teeth = smma(high, 8)  # Teeth (Red) - 8-period SMMA of median price
    lips = smma(high, 5)   # Lips (Green) - 5-period SMMA of median price
    
    # Use median price (typical price) for Alligator calculation
    median_price = (high + low + close) / 3
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # 1d EMA50 trend filter
        ema_long = close[i] > ema_50_1d_aligned[i]
        ema_short = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if alligator_long and ema_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif alligator_short and ema_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator reverses (Lips < Teeth) or trend reversal
            if lips[i] < teeth[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator reverses (Lips > Teeth) or trend reversal
            if lips[i] > teeth[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals