#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Bands breakout with 12-hour EMA50 filter and volume confirmation
# Long when price breaks above upper Bollinger Band(20,2) AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below lower Bollinger Band(20,2) AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Bollinger Bands (opposite band)
# Bollinger Bands capture volatility-based breakouts; EMA50 filters trend direction; volume confirms strength
# Designed for 4h timeframe with target 75-200 trades over 4 years to minimize fee drag

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
    
    # Calculate Bollinger Bands on 4h (20-period, 2 std dev)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper_band = (basis + 2 * dev).values
    lower_band = (basis - 2 * dev).values
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Bollinger + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above upper Bollinger Band + above 12h EMA50 + volume confirmation
            if (price > upper_band[i] and price > ema50_12h_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below lower Bollinger Band + below 12h EMA50 + volume confirmation
            elif (price < lower_band[i] and price < ema50_12h_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below lower Bollinger Band (opposite band)
            if price < lower_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above upper Bollinger Band (opposite band)
            if price > upper_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Bollinger_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0