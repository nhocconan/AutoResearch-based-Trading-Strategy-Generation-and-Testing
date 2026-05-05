#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND volume > 1.5x 20-period average AND 1w EMA50 uptrend
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND volume > 1.5x 20-period average AND 1w EMA50 downtrend
# Exit when Alligator alignment breaks OR 1w trend reverses
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-25 trades/year per symbol.
# Williams Alligator identifies trend phases via smoothed medians, volume confirms conviction, 1w EMA50 filters higher timeframe direction.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "1d_WilliamsAlligator_VolumeSpike_1wEMA50_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d data
    # Jaws: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(df_1d['high'].values, 13)  # Using high for jaws
    teeth = smma(df_1d['low'].values, 8)   # Using low for teeth
    lips = smma(df_1d['close'].values, 5)  # Using close for lips
    
    # Shift the lines: jaws shifted 8, teeth shifted 5, lips shifted 3
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Align Alligator lines to prices timeframe (1d to 1d = no shift needed, but use align_htf_to_ltf for safety)
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1w = close_1w > ema_50_1w
    downtrend_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)  # No volume confirmation if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaws_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish alignment (jaws < teeth < lips) AND price > lips AND volume spike AND 1w EMA50 uptrend
            if (jaws_aligned[i] < teeth_aligned[i] and 
                teeth_aligned[i] < lips_aligned[i] and 
                close[i] > lips_aligned[i] and 
                volume_filter[i] and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish alignment (jaws > teeth > lips) AND price < lips AND volume spike AND 1w EMA50 downtrend
            elif (jaws_aligned[i] > teeth_aligned[i] and 
                  teeth_aligned[i] > lips_aligned[i] and 
                  close[i] < lips_aligned[i] and 
                  volume_filter[i] and 
                  downtrend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR 1w trend changes to downtrend
            if not (jaws_aligned[i] < teeth_aligned[i] and 
                    teeth_aligned[i] < lips_aligned[i]) or \
               downtrend_1w_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR 1w trend changes to uptrend
            if not (jaws_aligned[i] > teeth_aligned[i] and 
                    teeth_aligned[i] > lips_aligned[i]) or \
               uptrend_1w_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals