#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band reversal with 1w EMA trend filter and volume confirmation.
# Long when price crosses below lower BB(20,2) with 1w EMA50 uptrend and volume > 1.5x average.
# Short when price crosses above upper BB(20,2) with 1w EMA50 downtrend and volume > 1.5x average.
# Exit when price returns to middle BB(20,2).
# Targets 10-25 trades per year, leveraging mean reversion in ranging markets while respecting weekly trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        sma[i] = np.mean(close[i - bb_period + 1:i + 1])
        std[i] = np.std(close[i - bb_period + 1:i + 1])
        middle[i] = sma[i]
        upper[i] = sma[i] + bb_std * std[i]
        lower[i] = sma[i] - bb_std * std[i]
    
    # Bollinger Bands previous values for crossover detection
    upper_prev = np.full(n, np.nan)
    lower_prev = np.full(n, np.nan)
    middle_prev = np.full(n, np.nan)
    upper_prev[1:] = upper[:-1]
    lower_prev[1:] = lower[:-1]
    middle_prev[1:] = middle[:-1]
    
    # Align 1w EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Bollinger Bands, EMA50, and volume MA20
    start_idx = max(bb_period, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(upper_prev[i]) or np.isnan(lower_prev[i]) or np.isnan(middle_prev[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price crosses below lower BB with 1w EMA50 uptrend and volume filter
            if (lower_prev[i] >= price and lower[i] < price and 
                price > ema_1w_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price crosses above upper BB with 1w EMA50 downtrend and volume filter
            elif (upper_prev[i] <= price and upper[i] > price and 
                  price < ema_1w_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB from below
            if middle_prev[i] <= price and middle[i] > price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle BB from above
            if middle_prev[i] >= price and middle[i] < price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Bollinger20_2_Reversal_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0