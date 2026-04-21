#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Supertrend for trend direction and 1h RSI mean reversion for entry timing.
In uptrend (Supertrend up), buy when RSI < 30 (oversold); in downtrend (Supertrend down), sell when RSI > 70 (overbought).
Supertrend filters for trend alignment, RSI provides mean-reversion entries within the trend.
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
Target: 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, multiplier=3.0)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first element NaN
    
    # ATR(10)
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl_avg = (high_4h + low_4h) / 2.0
    upper_band = hl_avg + 3.0 * atr
    lower_band = hl_avg - 3.0 * atr
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_4h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend direction to 1h timeframe
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # Calculate 1h RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.2x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(direction_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_dir = direction_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.2
        
        if position == 0:
            # Enter long: uptrend + RSI oversold + volume filter
            if (trend_dir == 1 and 
                rsi_val < 30 and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.20
                position = 1
            # Enter short: downtrend + RSI overbought + volume filter
            elif (trend_dir == -1 and 
                  rsi_val > 70 and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal
            if position == 1 and (rsi_val > 45 or trend_dir == -1):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi_val < 55 or trend_dir == 1):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Supertrend_RSI_MeanReversion"
timeframe = "1h"
leverage = 1.0