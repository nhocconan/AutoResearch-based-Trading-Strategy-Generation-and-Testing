#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 12-hour EMA50 trend filter and volume confirmation
# Williams Alligator: Jaw (13-period SMMA, 8-period shift), Teeth (8-period SMMA, 5-period shift), Lips (5-period SMMA, 3-period shift)
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw)
# Uses Alligator for trend identification, EMA for higher timeframe trend confirmation, volume for breakout validation
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def smma(data, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's smoothing"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=np.float64)
    result = np.full_like(data, np.nan, dtype=np.float64)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: (prev*(period-1) + current) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Williams Alligator components
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price = (high + low) / 2
    jaw_raw = smma(median_price, 13)
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = smma(median_price, 8)
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = smma(median_price, 5)
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need Alligator components)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Check Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long setup: bullish Alligator alignment + above 12h EMA50 + volume confirmation
            if bullish_alignment and price > ema50_12h_aligned[i] and vol > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short setup: bearish Alligator alignment + below 12h EMA50 + volume confirmation
            elif bearish_alignment and price < ema50_12h_aligned[i] and vol > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks (lips crosses below teeth)
            if lips[i] < teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator alignment breaks (lips crosses above teeth)
            if lips[i] > teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsAlligator_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0