#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
- Williams Alligator (Jaw/Teeth/Lips) identifies trendless markets and trend formation
- Only trade when Lips cross above/below Teeth with Jaw divergence (strong trend signal)
- 1w EMA(34) trend filter ensures alignment with higher timeframe direction
- Volume confirmation (> 2.0x 20-period average) filters weak breakouts
- Designed for 1d timeframe targeting 7-25 trades/year (30-100 over 4 years)
- Works in both bull and bear markets by trading with the 1w trend
- Alligator excels at catching trend starts after consolidation periods
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator: SMAs of median price
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    median_price_1d = (high_1d + low_1d) / 2.0
    
    # Smoothed Moving Average (SMMA) = EMA with alpha = 1/period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV * (period-1) + CURRENT) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_raw = smma(median_price_1d, 13)
    teeth_raw = smma(median_price_1d, 8)
    lips_raw = smma(median_price_1d, 5)
    
    # Apply shifts: Jaw(+8), Teeth(+5), Lips(+3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Invalidate shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 34)  # Alligator, volume MA, 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Alligator signals
        # Lips above Teeth AND Teeth above Jaw = bullish alignment
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] and 
                            teeth_aligned[i] > jaw_aligned[i])
        # Lips below Teeth AND Teeth below Jaw = bearish alignment
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i] and 
                            teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long conditions: bullish alignment + 1w uptrend + volume spike
            long_signal = (bullish_alignment and 
                          close[i] > ema_34_aligned[i] and
                          volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: bearish alignment + 1w downtrend + volume spike
            short_signal = (bearish_alignment and 
                           close[i] < ema_34_aligned[i] and
                           volume[i] > 2.0 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite alignment or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish alignment or price below 1w EMA
                if (bearish_alignment or 
                    close[i] < ema_34_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: bullish alignment or price above 1w EMA
                if (bullish_alignment or 
                    close[i] > ema_34_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA34_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0