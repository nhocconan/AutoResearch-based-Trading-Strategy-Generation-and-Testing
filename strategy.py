#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Long when Alligator jaws (13-period SMMA) < teeth (8-period SMMA) < lips (5-period SMMA) AND close > 1d EMA50 (bullish alignment)
- Short when Alligator jaws > teeth > lips AND close < 1d EMA50 (bearish alignment)
- Volume must be > 2.0 * median volume of last 20 bars (strong volume confirmation to avoid fakeouts)
- Exit on opposite Alligator alignment or trend reversal (close crosses 1d EMA50)
- Uses 4h primary timeframe with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Williams Alligator identifies trending vs ranging markets via jaw-teeth-lips alignment
- 1d EMA50 ensures alignment with daily trend to avoid whipsaws in counter-trend moves
- Strong volume confirmation (2.0x median) filters low-probability breakouts
- Designed for BTC/ETH with edge in trending markets where Alligator shows clear alignment
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length < 1:
        return source
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    result[length-1] = np.nanmean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components (SMMA of median price)
    median_price = (high + low) / 2
    jaws = smma(median_price, 13)   # Blue line (13-period)
    teeth = smma(median_price, 8)   # Red line (8-period)
    lips = smma(median_price, 5)    # Green line (5-period)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13) + 1  # 20 for volume, 13 for jaws (slowest)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = jaws[i] < teeth[i] < lips[i]   # Jaws < Teeth < Lips
        bearish_alignment = jaws[i] > teeth[i] > lips[i]   # Jaws > Teeth > Lips
        
        if position == 0:
            # Long: bullish Alligator alignment, trend up (close > EMA50), volume confirmation
            if bullish_alignment and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment, trend down (close < EMA50), volume confirmation
            elif bearish_alignment and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish Alligator alignment OR trend reversal (close < EMA50)
            if bearish_alignment or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish Alligator alignment OR trend reversal (close > EMA50)
            if bullish_alignment or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0