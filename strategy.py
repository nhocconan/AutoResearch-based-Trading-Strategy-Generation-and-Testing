#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams %R with 1-day trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold) AND price > 1-day EMA50 AND volume > 1.3x 20-period average
# Short when Williams %R crosses below -20 (overbought) AND price < 1-day EMA50 AND volume > 1.3x 20-period average
# Exit when Williams %R crosses back through -50 (mean reversion)
# Williams %R captures momentum extremes, EMA50 provides trend filter, volume confirms conviction
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average for confirmation (20-period on 1d)
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Williams %R + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        williams_r_val = williams_r_aligned[i]
        vol_threshold = vol_avg_1d_aligned[i] * 1.3
        
        if position == 0:
            # Long setup: Williams %R crosses above -80 + above 1d EMA50 + volume confirmation
            if (williams_r_val > -80 and williams_r_aligned[i-1] <= -80 and 
                price > ema50_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R crosses below -20 + below 1d EMA50 + volume confirmation
            elif (williams_r_val < -20 and williams_r_aligned[i-1] >= -20 and 
                  price < ema50_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (overbought territory)
            if williams_r_val > -50 and williams_r_aligned[i-1] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (oversold territory)
            if williams_r_val < -50 and williams_r_aligned[i-1] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_WilliamsR_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0