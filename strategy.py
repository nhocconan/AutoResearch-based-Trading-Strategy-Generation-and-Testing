#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour price action filtered by weekly trend and daily momentum
# Long when: price > weekly EMA200 AND daily RSI(14) > 50 AND price breaks above 6h Donchian(10) high
# Short when: price < weekly EMA200 AND daily RSI(14) < 50 AND price breaks below 6h Donchian(10) low
# Exit when price crosses the opposite Donchian band
# Uses weekly trend filter (EMA200) to avoid counter-trend trades, daily RSI for momentum confirmation
# Donchian(10) on 6h provides timely breakouts with moderate frequency
# Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    # Load daily data ONCE before loop for momentum filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate daily RSI(14) for momentum filter
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Calculate Donchian channels on 6h (10-period high/low)
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(200, 14, 10) + 5
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(high_10[i]) or np.isnan(low_10[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: above weekly EMA200 AND daily RSI > 50 AND breakout above Donchian high
            if (price > ema200_1w_aligned[i] and rsi_1d_aligned[i] > 50 and price > high_10[i]):
                position = 1
                signals[i] = position_size
            # Short setup: below weekly EMA200 AND daily RSI < 50 AND breakdown below Donchian low
            elif (price < ema200_1w_aligned[i] and rsi_1d_aligned[i] < 50 and price < low_10[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below Donchian low (opposite band)
            if price < low_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above Donchian high (opposite band)
            if price > high_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WeeklyEMA200_DailyRSI_Donchian10"
timeframe = "6h"
leverage = 1.0