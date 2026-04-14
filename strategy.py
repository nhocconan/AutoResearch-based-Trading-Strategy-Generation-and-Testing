#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day EMA filter and volume confirmation
# Long when price above Kumo (cloud) AND Tenkan > Kijun AND price > daily EMA50 AND volume > 1.5x 20-period average
# Short when price below Kumo (cloud) AND Tenkan < Kijun AND price < daily EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses back into Kumo or Tenkan/Kijun cross reverses
# Ichimoku provides multi-timeframe support/resistance, EMA50 filters trend, volume confirms strength
# Target: 60-150 total trades over 4 years (15-38/year) for balanced opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components on 6h
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
    
    # Load daily data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
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
    start = 55
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Kumo (cloud) boundaries - Senkou Span A and B shifted 26 periods ahead
        # For current price, we use the Senkou values from 26 periods ago
        senkou_a_current = senkou_a[i-26] if i >= 26 else senkou_a[i]
        senkou_b_current = senkou_b[i-26] if i >= 26 else senkou_b[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_current, senkou_b_current)
        cloud_bottom = min(senkou_a_current, senkou_b_current)
        
        if position == 0:
            # Long setup: price above cloud AND Tenkan > Kijun AND price > daily EMA50 AND volume confirmation
            if (price > cloud_top and tenkan[i] > kijun[i] and 
                price > ema50_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price below cloud AND Tenkan < Kijun AND price < daily EMA50 AND volume confirmation
            elif (price < cloud_bottom and tenkan[i] < kijun[i] and 
                  price < ema50_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below cloud OR Tenkan/Kijun cross reverses
            if price < cloud_bottom or tenkan[i] < kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above cloud OR Tenkan/Kijun cross reverses
            if price > cloud_top or tenkan[i] > kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Ichimoku_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0