#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-day EMA50 trend filter and volume confirmation
# Long when green line > red line > blue line AND price > 1d EMA50 AND volume > 1.5x 20-period average
# Short when green line < red line < blue line AND price < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when lines re-cross (green crosses red)
# Williams Alligator identifies trend phases; EMA50 filters for higher-timeframe trend; volume confirms strength
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator lines (13,8,5 smoothed with future shift)
    # Jaw (blue): 13-period SMMA, shifted 8 bars
    # Teeth (red): 8-period SMMA, shifted 5 bars
    # Lips (green): 5-period SMMA, shifted 3 bars
    def smoothed_moving_average(values, period):
        sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
        # SMMA: first value = SMA, then SMMA = (prev*(period-1) + current) / period
        smma = np.full_like(values, np.nan, dtype=float)
        if len(values) >= period:
            smma[period-1] = sma[period-1]
            for i in range(period, len(values)):
                smma[i] = (smma[i-1] * (period-1) + values[i]) / period
        return smma
    
    jaw = smoothed_moving_average(close, 13)
    teeth = smoothed_moving_average(close, 8)
    lips = smoothed_moving_average(close, 5)
    
    # Shift jaws: 8, teeth: 5, lips: 3 (Alligator specific)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set first values to nan where roll brings in old data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max period + shifts)
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: lips > teeth > jaw (green > red > blue) AND price > 1d EMA50 AND volume confirmation
            if (lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i] and 
                price > ema50_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: lips < teeth < jaw (green < red < blue) AND price < 1d EMA50 AND volume confirmation
            elif (lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i] and 
                  price < ema50_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: lips crosses below teeth (green crosses below red)
            if lips_shifted[i] < teeth_shifted[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: lips crosses above teeth (green crosses above red)
            if lips_shifted[i] > teeth_shifted[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0