#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3)
# Long when Lips > Teeth > Jaw (bullish alignment) AND 1d close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when Lips < Teeth < Jaw (bearish alignment) AND 1d close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw)
# Uses 12h primary timeframe with 1d HTF for trend filter and Alligator structure
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) based on proven Williams Alligator performance
# Williams Alligator identifies trend phases; 1d EMA50 filters for higher-timeframe trend; volume confirms breakout validity
# Works in both bull and bear markets by following the 1d trend while using 12h for entry timing

name = "12h_Williams_Alligator_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def smma(source, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's MA or RMA"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    result = np.empty_like(source)
    result[:] = np.nan
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for trend filter and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d close
    # Jaw: 13-period SMMA smoothed 8
    jaw_raw = smma(df_1d['close'].values, 13)
    jaw = smma(jaw_raw, 8)
    # Teeth: 8-period SMMA smoothed 5
    teeth_raw = smma(df_1d['close'].values, 8)
    teeth = smma(teeth_raw, 5)
    # Lips: 5-period SMMA smoothed 3
    lips_raw = smma(df_1d['close'].values, 5)
    lips = smma(lips_raw, 3)
    
    # Align Williams Alligator lines to 12h timeframe (wait for 1d bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND 1d close > 1d EMA50 AND volume spike
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND 1d close < 1d EMA50 AND volume spike
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips crosses below Teeth OR Teeth crosses below Jaw)
            if (lips_aligned[i] < teeth_aligned[i] or 
                teeth_aligned[i] < jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips crosses above Teeth OR Teeth crosses above Jaw)
            if (lips_aligned[i] > teeth_aligned[i] or 
                teeth_aligned[i] > jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals