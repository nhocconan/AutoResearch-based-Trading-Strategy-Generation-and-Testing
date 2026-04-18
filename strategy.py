#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with daily trend filter and volume confirmation.
# Uses daily EMA13 for trend direction, Alligator lines for entry signals, and volume filter.
# Designed for low trade frequency (~20-40/year) to minimize fee drag while capturing trends.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA13 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA(13) on daily close for trend direction
    close_1d_series = pd.Series(close_1d)
    ema_13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align daily EMA13 to 12h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: (prev*(period-1) + current) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Calculate 12h ATR for position sizing and stops
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need daily EMA13, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: price above daily EMA13 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_13_1d_aligned[i]
        trend_down = close[i] < ema_13_1d_aligned[i]
        
        # Alligator signals: Lips above Teeth above Jaw = uptrend
        # Lips below Teeth below Jaw = downtrend
        alligator_long = (lips_shifted[i] > teeth_shifted[i] and 
                         teeth_shifted[i] > jaw_shifted[i])
        alligator_short = (lips_shifted[i] < teeth_shifted[i] and 
                          teeth_shifted[i] < jaw_shifted[i])
        
        if position == 0:
            # Long entry: Alligator bullish alignment + volume + trend filter
            if (alligator_long and vol_confirmed and trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish alignment + volume + trend filter
            elif (alligator_short and vol_confirmed and trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: Alligator bearish crossover or ATR stop
            if (lips_shifted[i] < teeth_shifted[i] or 
                close[i] < open_price[i] - 2.0 * atr_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator bullish crossover or ATR stop
            if (lips_shifted[i] > teeth_shifted[i] or 
                close[i] > open_price[i] + 2.0 * atr_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_DailyEMA13_VolumeFilter"
timeframe = "12h"
leverage = 1.0