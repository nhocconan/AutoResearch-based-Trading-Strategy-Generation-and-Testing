#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot reversal with weekly trend filter and volume confirmation
# Long when price touches Camarilla L3 (support) AND price > weekly EMA50 AND volume > 1.5x 20-period average
# Short when price touches Camarilla H3 (resistance) AND price < weekly EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses opposite H3/L3 level or reverses to Camarilla pivot point
# This captures mean-reversion bounces at strong intraday levels with trend alignment
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Calculate Camarilla levels using previous day's OHLC
        # Need daily OHLC for Camarilla calculation
        if i < 1:  # Need at least one day of data
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC (assuming 12h timeframe, 2 bars per day)
        prev_day_idx = i - 2
        if prev_day_idx < 0:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels for current day based on previous day
        ph = high[prev_day_idx]  # Previous day high
        pl = low[prev_day_idx]   # Previous day low
        pc = close[prev_day_idx] # Previous day close
        
        # Camarilla levels
        range_val = ph - pl
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Resistance levels
        h3 = pc + (range_val * 1.1 / 4)  # H3 = C + 1.1*(H-L)/4
        h4 = pc + (range_val * 1.1 / 2)  # H4 = C + 1.1*(H-L)/2
        # Support levels
        l3 = pc - (range_val * 1.1 / 4)  # L3 = C - 1.1*(H-L)/4
        l4 = pc - (range_val * 1.1 / 2)  # L4 = C - 1.1*(H-L)/2
        # Pivot point
        pp = (ph + pl + pc) / 3
        
        if position == 0:
            # Long setup: price touches L3 support AND above weekly EMA50 AND volume confirmation
            if (abs(price - l3) < 0.001 * price and price > ema50_1w_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price touches H3 resistance AND below weekly EMA50 AND volume confirmation
            elif (abs(price - h3) < 0.001 * price and price < ema50_1w_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above H3 or returns to pivot point
            if price > h3 or abs(price - pp) < 0.0005 * price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below L3 or returns to pivot point
            if price < l3 or abs(price - pp) < 0.0005 * price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0