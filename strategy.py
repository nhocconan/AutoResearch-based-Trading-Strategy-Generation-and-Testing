#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + volume confirmation + 1w ADX trend filter
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and entries when
# Lips cross Jaw/Teeth in trending markets (ADX > 25 on 1w). Volume confirms momentum.
# Designed for low-frequency, high-conviction trades on 12h timeframe to avoid fee drag.
# Works in bull/bear by only taking signals aligned with weekly trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator on 12h
    # Jaw (Blue): 13-period SMMA, shifted 8 bars
    # Teeth (Red): 8-period SMMA, shifted 5 bars
    # Lips (Green): 5-period SMMA, shifted 3 bars
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = np.full_like(data, np.nan, dtype=float)
        sma[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Calculate ADX (14-period) on 1w for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        # Smoothed TR, +DM, -DM
        atr = np.full_like(tr, np.nan, dtype=float)
        plus_dm_smooth = np.full_like(tr, np.nan, dtype=float)
        minus_dm_smooth = np.full_like(tr, np.nan, dtype=float)
        
        atr[period-1] = np.mean(tr[:period])
        plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.full_like(dx, np.nan, dtype=float)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align all indicators to 12h timeframe (LTF)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size: 25% of capital
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            continue
        
        # Long entry: Lips cross above Jaw/Teeth + ADX > 25 (trending) + volume confirmation
        if (lips_aligned[i] > jaw_aligned[i] and lips_aligned[i] > teeth_aligned[i] and
            lips_aligned[i-1] <= jaw_aligned[i-1] and lips_aligned[i-1] <= teeth_aligned[i-1] and
            adx_1w_aligned[i] > 25 and
            volume[i] > 1.5 * np.nanmedian(volume[max(0, i-20):i]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Lips cross below Jaw/Teeth + ADX > 25 (trending) + volume confirmation
        elif (lips_aligned[i] < jaw_aligned[i] and lips_aligned[i] < teeth_aligned[i] and
              lips_aligned[i-1] >= jaw_aligned[i-1] and lips_aligned[i-1] >= teeth_aligned[i-1] and
              adx_1w_aligned[i] > 25 and
              volume[i] > 1.5 * np.nanmedian(volume[max(0, i-20):i]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Lips cross back inside Jaw/Teeth or ADX weakens (< 20)
        elif position == 1 and (lips_aligned[i] < jaw_aligned[i] or lips_aligned[i] < teeth_aligned[i] or adx_1w_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (lips_aligned[i] > jaw_aligned[i] or lips_aligned[i] > teeth_aligned[i] or adx_1w_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_ADX_Volume"
timeframe = "12h"
leverage = 1.0