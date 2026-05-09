#!/usr/bin/env python3
# Hypothesis: 1h price action with 4h trend filter and volume confirmation
# Long when price > 4h EMA20, RSI(14) > 50, and volume > 1.3x 20-period average
# Short when price < 4h EMA20, RSI(14) < 50, and volume > 1.3x 20-period average
# Exit when price crosses back across 4h EMA20
# Uses 4h EMA20 for trend direction to filter noise, RSI for momentum, volume for confirmation
# Designed for 1h timeframe to capture medium-term moves with controlled trade frequency
# Target: 100-150 total trades over 4 years (25-38/year) with size 0.20

name = "1h_EMA20_RSI_Volume_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.3 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for RSI and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above 4h EMA20, RSI > 50, volume confirmation
            if (close[i] > ema20_4h_aligned[i] and 
                rsi[i] > 50 and 
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price below 4h EMA20, RSI < 50, volume confirmation
            elif (close[i] < ema20_4h_aligned[i] and 
                  rsi[i] < 50 and 
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 4h EMA20
            if close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above 4h EMA20
            if close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals