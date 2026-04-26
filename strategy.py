#!/usr/bin/env python3
"""
1d_WilliamsAlligator_1wTrend_v1
Hypothesis: Trade Williams Alligator signals on 1d timeframe aligned with 1w EMA50 trend filter.
Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
Only trade when Alligator is "awake" (lines intertwined) and price is outside the Alligator mouth,
in the direction of the 1w EMA50 trend. Volume confirmation reduces false signals.
Designed for low trade frequency (7-25/year) to minimize fee drag and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(data, period):
        """Smoothed Moving Average"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    close_1d = df_1d['close'].values
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply shifts (Jaw: 8, Teeth: 5, Lips: 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # First shifted values remain NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Get 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 1.5x median volume (20-period) on 1d
    vol_median_1d = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops on 1d
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First period
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Alligator periods (13+8), EMA (50), volume median (20), ATR (14)
    start_idx = max(13+8, 50, 20, 14)  # 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_median_1d[i]) or np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median_1d[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        atr_val = atr[i]
        
        # Alligator "awake" condition: lines are intertwined (not all same direction)
        # Simplified: check if not (all rising or all falling)
        all_rising = lips_val > teeth_val > jaw_val
        all_falling = lips_val < teeth_val < jaw_val
        alligator_awake = not (all_rising or all_falling)
        
        if position == 0:
            # Long: price above Alligator mouth (lips) + uptrend + volume + Alligator awake
            long_signal = (close_val > lips_val) and \
                          (close_val > ema_50_1w_val) and \
                          (volume_val > 1.5 * vol_median_val) and \
                          alligator_awake
            
            # Short: price below Alligator mouth (jaws) + downtrend + volume + Alligator awake
            short_signal = (close_val < jaw_val) and \
                           (close_val < ema_50_1w_val) and \
                           (volume_val > 1.5 * vol_median_val) and \
                           alligator_awake
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            # 1. Price breaks below teeth (reversal signal)
            # 2. Trend changes (close < 1w EMA50)
            # 3. Alligator starts sleeping (all lines aligned)
            # 4. ATR-based stop loss (2.5 * ATR below entry)
            # 5. Profit target (4.0 * ATR above entry)
            if (close_val < teeth_val) or \
               (close_val < ema_50_1w_val) or \
               (not alligator_awake) or \
               (close_val < entry_price - 2.5 * atr_val) or \
               (close_val > entry_price + 4.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            # 1. Price breaks above teeth (reversal signal)
            # 2. Trend changes (close > 1w EMA50)
            # 3. Alligator starts sleeping (all lines aligned)
            # 4. ATR-based stop loss (2.5 * ATR above entry)
            # 5. Profit target (4.0 * ATR below entry)
            if (close_val > teeth_val) or \
               (close_val > ema_50_1w_val) or \
               (not alligator_awake) or \
               (close_val > entry_price + 2.5 * atr_val) or \
               (close_val < entry_price - 4.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsAlligator_1wTrend_v1"
timeframe = "1d"
leverage = 1.0