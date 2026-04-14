#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud breakout with 1-day trend filter and volume confirmation
# Long when price breaks above Kumo (cloud) + Tenkan > Kijun + price > daily EMA50 + volume > 1.5x 20-period average
# Short when price breaks below Kumo + Tenkan < Kijun + price < daily EMA50 + volume > 1.5x 20-period average
# Exit when price crosses back inside the Kumo (opposite side)
# This captures strong momentum with Ichimoku's multi-line confirmation while avoiding false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Kumo top and bottom (cloud)
    kumotop = np.maximum(senkou_a, senkou_b)
    kumbottom = np.minimum(senkou_a, senkou_b)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (52 for Senkou B)
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumotop[i]) or np.isnan(kumbottom[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price above cloud + Tenkan > Kijun + above daily EMA50 + volume confirmation
            if (price > kumotop[i] and tenkan[i] > kijun[i] and 
                price > ema50_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price below cloud + Tenkan < Kijun + below daily EMA50 + volume confirmation
            elif (price < kumbottom[i] and tenkan[i] < kijun[i] and 
                  price < ema50_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below cloud bottom (Kumo bottom)
            if price < kumbottom[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above cloud top (Kumo top)
            if price > kumotop[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Ichimoku_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0