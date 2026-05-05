#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1w EMA50 trend filter + volume confirmation
# Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3)
# Long when Lips > Teeth > Jaw (bullish alignment) AND close > 1w EMA50 AND volume > 2.0x 20-period average
# Short when Lips < Teeth < Jaw (bearish alignment) AND close < 1w EMA50 AND volume > 2.0x 20-period average
# Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw) OR close crosses 1w EMA50
# Uses 12h primary timeframe with 1w HTF for trend filter to capture sustained multi-week moves
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Alligator identifies trend initiation and strength; 1w EMA50 filters for higher-timeframe trend; volume confirms participation

name = "12h_Williams_Alligator_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator components on 12h data using SMMA (Smoothed Moving Average)
    def smma(arr, period):
        """Smoothed Moving Average - similar to Wilder's smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw_raw = smma(close, 13)
    jaw = smma(jaw_raw, 8)  # Smoothed further by 8
    
    teeth_raw = smma(close, 8)
    teeth = smma(teeth_raw, 5)  # Smoothed further by 5
    
    lips_raw = smma(close, 5)
    lips = smma(lips_raw, 3)  # Smoothed further by 3
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) 
            #              AND close > 1w EMA50 AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment)
            #               AND close < 1w EMA50 AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR close < 1w EMA50 (trend flip)
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR close > 1w EMA50 (trend flip)
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals