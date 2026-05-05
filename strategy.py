#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for trend direction and 1h RSI(14) for entry timing
# Long when 4h Supertrend is bullish AND 1h RSI(14) crosses above 30 (oversold bounce)
# Short when 4h Supertrend is bearish AND 1h RSI(14) crosses below 70 (overbought rejection)
# Exit when RSI crosses 50 in opposite direction or Supertrend flips
# Uses discrete sizing 0.20 to minimize fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h
# Supertrend on 4h provides robust trend filtering that works in both bull and bear markets
# RSI(14) on 1h provides precise entry timing for mean reversion within the trend
# Session filter (08-20 UTC) reduces noise during low-volume periods

name = "1h_4hSupertrend_1hRSI_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data ONCE before loop for Supertrend calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:  # Need sufficient data for ATR
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) for 4h Supertrend
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.max([high_4h[0] - low_4h[0], np.abs(high_4h[0] - close_4h[0]), np.abs(low_4h[0] - close_4h[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend components
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    # Initialize Supertrend arrays
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(1, len(close_4h)):
        if close_4h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align 4h Supertrend to 1h timeframe (wait for completed 4h bar)
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(rsi[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h Supertrend bullish AND 1h RSI crosses above 30 (oversold bounce)
            if (direction_aligned[i] == 1 and 
                rsi[i] > 30 and rsi[i-1] <= 30):
                signals[i] = 0.20
                position = 1
            # Short: 4h Supertrend bearish AND 1h RSI crosses below 70 (overbought rejection)
            elif (direction_aligned[i] == -1 and 
                  rsi[i] < 70 and rsi[i-1] >= 70):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI crosses below 50 OR Supertrend flips bearish
            if (rsi[i] < 50 and rsi[i-1] >= 50) or direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI crosses above 50 OR Supertrend flips bullish
            if (rsi[i] > 50 and rsi[i-1] <= 50) or direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals