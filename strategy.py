#!/usr/bin/env python3
"""
4h_1d_rsi_congestion_filter_v1
Hypothesis: 4-hour strategy using daily RSI to filter congestion periods and 4-hour price action for entries.
Enters long when RSI < 30 (oversold) and price breaks above 4-hour high of last 20 bars with volume confirmation.
Enters short when RSI > 70 (overbought) and price breaks below 4-hour low of last 20 bars with volume confirmation.
Uses congestion filter: only trade when daily RSI is in extreme zones to avoid whipsaws in ranging markets.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drift while capturing mean-reversion moves in extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI congestion filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI (14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 4-hour rolling high/low for breakout detection
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Congestion filter: only trade in extreme RSI zones
        oversold = rsi_1d_aligned[i] < 30
        overbought = rsi_1d_aligned[i] > 70
        
        # Breakout conditions
        bullish_breakout = close[i] > high_20[i-1]  # Break above recent high
        bearish_breakout = close[i] < low_20[i-1]   # Break below recent low
        
        # Entry conditions
        long_entry = oversold and bullish_breakout and volume_filter
        short_entry = overbought and bearish_breakout and volume_filter
        
        # Exit conditions: opposite RSI extreme or opposite breakout
        long_exit = rsi_1d_aligned[i] > 70 or close[i] < low_20[i-1]
        short_exit = rsi_1d_aligned[i] < 30 or close[i] > high_20[i-1]
        
        # Fixed position size to minimize churn
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_rsi_congestion_filter_v1"
timeframe = "4h"
leverage = 1.0