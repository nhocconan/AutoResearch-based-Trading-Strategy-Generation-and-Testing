#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 12-hour trend filter and volume confirmation
# Long when price > Alligator Jaw (13-period smoothed median) AND Alligator Mouth is open (Jaw < Teeth < Lips) AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when price < Alligator Jaw AND Alligator Mouth is open (Lips < Teeth < Jaw) AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Alligator Jaw (opposite condition)
# Williams Alligator identifies trend direction and strength; 12h EMA50 filters for higher timeframe trend; volume confirms momentum
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components on 6h (13, 8, 5 periods with smoothing)
    median = (high + low) / 2
    
    # Jaw: 13-period SMMA of median, shifted 8 bars
    jaw_raw = pd.Series(median).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    
    # Teeth: 8-period SMMA of median, shifted 5 bars
    teeth_raw = pd.Series(median).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    
    # Lips: 5-period SMMA of median, shifted 3 bars
    lips_raw = pd.Series(median).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max shift + periods)
    start = 25
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Check if Alligator Mouth is open (trending condition)
        jaw_open = jaw[i] < teeth[i] < lips[i]  # bullish alignment
        teeth_open = lips[i] < teeth[i] < jaw[i]  # bearish alignment
        
        if position == 0:
            # Long setup: price > Jaw AND Jaw < Teeth < Lips (bullish alignment) AND price > 12h EMA50 AND volume confirmation
            if (price > jaw[i] and jaw_open and price > ema50_12h_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price < Jaw AND Lips < Teeth < Jaw (bearish alignment) AND price < 12h EMA50 AND volume confirmation
            elif (price < jaw[i] and teeth_open and price < ema50_12h_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below Jaw OR Alligator closes mouth (loss of trend)
            if price < jaw[i] or not jaw_open:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above Jaw OR Alligator closes mouth (loss of trend)
            if price > jaw[i] or not teeth_open:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsAlligator_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0