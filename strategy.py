#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA(34) trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) identifies trend absence when lines are intertwined.
# Breakout occurs when Lips cross above/below Jaw with Teeth confirming direction.
# 1d EMA(34) ensures we trade with higher timeframe trend to avoid counter-trend whipsaws.
# Volume confirmation filters false breakouts. Designed for low trade frequency (12-37/year) to minimize fee drag.
# Works in both bull (trend continuation) and bear (trend reversal) markets by capturing strong moves.

name = "12h_WilliamsAlligator_1dEMA34_Trend_Volume_v1"
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
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h
    # Jaw: 13-period SMMA (smoothed) of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) - similar to Wilder's smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=np.float64)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Invalidate the shifted values (set to NaN where shift creates invalid data)
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe (no additional delay needed - SMMA uses completed bars)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw) if len(df_1d) >= 2 else jaw
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth) if len(df_1d) >= 2 else teeth
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips) if len(df_1d) >= 2 else lips
    
    # Volume confirmation (2.0x 20-period average) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(34 for 1d EMA, 13+8 for Alligator shifts, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Lips above Jaw AND Teeth above Jaw (bullish alignment) + price above EMA + volume spike
            if (lips_aligned[i] > jaw_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Lips below Jaw AND Teeth below Jaw (bearish alignment) + price below EMA + volume spike
            elif (lips_aligned[i] < jaw_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Lips cross below Jaw (trend weakness) or price below EMA (trend reversal)
            if lips_aligned[i] < jaw_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Lips cross above Jaw (trend weakness) or price above EMA (trend reversal)
            if lips_aligned[i] > jaw_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals