#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R with 12-hour trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 12h EMA200 AND volume > 1.5x 20-period average
# Short when Williams %R > -20 (overbought) AND price < 12h EMA200 AND volume > 1.5x 20-period average
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# Williams %R identifies overextended moves likely to reverse, EMA200 ensures trend alignment,
# volume confirmation avoids false signals. Designed for mean reversion in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Williams %R (14-period) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (14 for Williams %R + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema200_12h_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Williams %R oversold (< -80) + above 12h EMA200 + volume confirmation
            if (wr < -80 and price > ema200_12h_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R overbought (> -20) + below 12h EMA200 + volume confirmation
            elif (wr > -20 and price < ema200_12h_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses back above -50
            if wr > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses back below -50
            if wr < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_12hEMA200_Volume"
timeframe = "4h"
leverage = 1.0