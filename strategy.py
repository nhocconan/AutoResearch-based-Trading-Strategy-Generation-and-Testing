#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-week EMA50 trend filter and volume confirmation
# Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period smoothed median), Lips (5-period smoothed median)
# Long when Lips > Teeth > Jaw AND price > 1-week EMA50 AND volume > 1.5x 20-period average
# Short when Lips < Teeth < Jaw AND price < 1-week EMA50 AND volume > 1.5x 20-period average
# Exit when Lips crosses back through Teeth (Alligator "sleeping" signal)
# Uses Alligator for trend identification, weekly EMA for higher timeframe trend filter, volume for confirmation
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-week data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams Alligator components (using median price)
    median_price = (high + low) / 2
    median_series = pd.Series(median_price)
    
    # Jaw: 13-period smoothed median, 8 periods forward
    jaw_raw = median_series.rolling(window=13, min_periods=13).median()
    jaw = jaw_raw.shift(8)  # 8 periods forward displacement
    
    # Teeth: 8-period smoothed median, 5 periods forward
    teeth_raw = median_series.rolling(window=8, min_periods=8).median()
    teeth = teeth_raw.shift(5)  # 5 periods forward displacement
    
    # Lips: 5-period smoothed median, 3 periods forward
    lips_raw = median_series.rolling(window=5, min_periods=5).median()
    lips = lips_raw.shift(3)  # 3 periods forward displacement
    
    # Calculate 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max displacement + buffer)
    start = 25
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Lips > Teeth > Jaw (Alligator awake, bullish) AND price > 1w EMA50 AND volume confirmation
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                price > ema50_1w_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Lips < Teeth < Jaw (Alligator awake, bearish) AND price < 1w EMA50 AND volume confirmation
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  price < ema50_1w_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Lips crosses back below Teeth (Alligator going to sleep)
            if lips[i] < teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Lips crosses back above Teeth (Alligator going to sleep)
            if lips[i] > teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsAlligator_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0