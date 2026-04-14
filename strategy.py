#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour KAMA (Kaufman Adaptive Moving Average) trend filter with 1-day Bollinger Band mean reversion
# Long when price crosses below lower Bollinger Band (20,2) on 1d AND price > KAMA(10) on 4h
# Short when price crosses above upper Bollinger Band (20,2) on 1d AND price < KAMA(10) on 4h
# Exit when price crosses back inside the Bollinger Bands
# Uses KAMA for adaptive trend following and Bollinger Bands for mean reversion extremes
# Works in bull markets (trend filter) and bear markets (mean reversion at extremes)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE before loop for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA(10) on 4h for adaptive trend filter
    # ER = |net change| / sum(|changes|)
    # SC = [ER * (fastest - slowest) + slowest]^2
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    direction = abs(close_series - close_series.shift(10))
    er = direction / volatility.replace(0, np.nan)
    fast_sc = 2 / (2 + 1)  # 2-period EMA
    slow_sc = 2 / (30 + 1)  # 30-period EMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate Bollinger Bands on 1-day (20-period, 2 std dev)
    close_1d = df_1d['close'].values
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + 2 * std20_1d
    lower_bb_1d = sma20_1d - 2 * std20_1d
    
    # Align 1-day Bollinger Bands to 4h timeframe
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(upper_bb_1d_aligned[i]) or 
            np.isnan(lower_bb_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: price below lower 1d BB AND price above KAMA (mean reversion in uptrend)
            if (price < lower_bb_1d_aligned[i] and price > kama[i]):
                position = 1
                signals[i] = position_size
            # Short setup: price above upper 1d BB AND price below KAMA (mean reversion in downtrend)
            elif (price > upper_bb_1d_aligned[i] and price < kama[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back above lower Bollinger Band (mean reversion complete)
            if price > lower_bb_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back below upper Bollinger Band (mean reversion complete)
            if price < upper_bb_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_KAMA_1dBollinger_MeanReversion"
timeframe = "4h"
leverage = 1.0