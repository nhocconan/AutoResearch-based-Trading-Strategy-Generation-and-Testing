#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (13,8,5) with 1d trend filter and volume spike
# Uses Williams Alligator to identify trends (Jaw=13, Teeth=8, Lips=5)
# Long when Lips > Teeth > Jaw (bullish alignment) + price above Teeth + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + price below Teeth + volume spike
# Trend filter: 1d EMA50 confirms higher timeframe direction
# Designed to work in both bull and bear markets by following 1d trend
# Target: 20-40 trades/year to minimize fee drag while capturing significant moves

name = "4h_Williams_Alligator_1dTrend_VolumeSpike"
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
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 4h data
    # Jaw (blue line): 13-period SMMA smoothed 8 bars ahead
    # Teeth (red line): 8-period SMMA smoothed 5 bars ahead  
    # Lips (green line): 5-period SMMA smoothed 3 bars ahead
    
    def smma(series, period):
        """Smoothed Moving Average"""
        if len(series) < period:
            return np.full_like(series, np.nan)
        result = np.full_like(series, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(series[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    # Calculate SMMA for different periods
    smma_13 = smma(close, 13)
    smma_8 = smma(close, 8)
    smma_5 = smma(close, 5)
    
    # Alligator lines with smoothing offsets
    jaw = np.roll(smma_13, 8)   # Jaw: SMMA(13) shifted 8 bars ahead
    teeth = np.roll(smma_8, 5)  # Teeth: SMMA(8) shifted 5 bars ahead
    lips = np.roll(smma_5, 3)   # Lips: SMMA(5) shifted 3 bars ahead
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bullish alignment (Lips > Teeth > Jaw) + price above Teeth + volume spike + uptrend
            if (lips_val > teeth_val and teeth_val > jaw_val and 
                close[i] > teeth_val and 
                vol_spike and 
                close[i] > ema50_1d_val):
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish alignment (Lips < Teeth < Jaw) + price below Teeth + volume spike + downtrend
            elif (lips_val < teeth_val and teeth_val < jaw_val and 
                  close[i] < teeth_val and 
                  vol_spike and 
                  close[i] < ema50_1d_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish alignment OR price below Teeth OR trend turns down
            if (lips_val < teeth_val or teeth_val < jaw_val or 
                close[i] < teeth_val or 
                close[i] < ema50_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish alignment OR price above Teeth OR trend turns up
            if (lips_val > teeth_val or teeth_val > jaw_val or 
                close[i] > teeth_val or 
                close[i] > ema50_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals