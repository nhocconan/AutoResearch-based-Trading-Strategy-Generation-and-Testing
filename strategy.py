#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator with 12-hour EMA trend filter and volume confirmation.
# The Williams Alligator uses three SMAs (jaw, teeth, lips) to detect trends.
# We use the 12-hour EMA(50) as a higher-timeframe trend filter to avoid counter-trend trades.
# Entry occurs when the Alligator lines are aligned (lips > teeth > jaw for long, opposite for short)
# AND price confirms in the direction of the 12-hour EMA trend.
# Volume > 1.3x the 20-period average confirms participation.
# Exit when the Alligator lines become misaligned (trend weakness) or price crosses the 8-period EMA.
# Designed for 20-35 trades per year per symbol (80-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(50) for trend filter
    ema_len = 50
    if len(df_12h) < ema_len:
        return np.zeros(n)
    
    ema_12h = pd.Series(df_12h['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Williams Alligator components (13, 8, 5 SMAs with future shifts)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate SMMA (smoothed moving average) - equivalent to RMA/Wilder's smoothing
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, jaw_period)  # Using high for jaw as per original Alligator
    teeth = smma(low, teeth_period)  # Using low for teeth
    lips = smma(close, lips_period)  # Using close for lips
    
    # Apply shifts (shift forward means we use future data, so we need to lag)
    jaw_shifted = np.roll(jaw, -jaw_shift)
    teeth_shifted = np.roll(teeth, -teeth_shift)
    lips_shifted = np.roll(lips, -lips_shift)
    
    # Set shifted values to NaN where they would look ahead
    jaw_shifted[-jaw_shift:] = np.nan
    teeth_shifted[-teeth_shift:] = np.nan
    lips_shifted[-lips_shift:] = np.nan
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 8-period EMA for exit signal
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 50, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or
            np.isnan(lips_shifted[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(ema8[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: lips > teeth > jaw = bullish, lips < teeth < jaw = bearish
        lips = lips_shifted[i]
        teeth = teeth_shifted[i]
        jaw = jaw_shifted[i]
        
        bullish_aligned = (lips > teeth) and (teeth > jaw)
        bearish_aligned = (lips < teeth) and (teeth < jaw)
        
        # Trend filter: price relative to 12h EMA50
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Alligator bullish + above 12h EMA + volume
            if bullish_aligned and above_ema and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: Alligator bearish + below 12h EMA + volume
            elif bearish_aligned and below_ema and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator loses alignment OR price crosses below 8 EMA
            if not bullish_aligned or close[i] < ema8[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator loses alignment OR price crosses above 8 EMA
            if not bearish_aligned or close[i] > ema8[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_EMA50_Alligator_Volume_v1"
timeframe = "4h"
leverage = 1.0