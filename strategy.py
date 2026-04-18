#!/usr/bin/env python3
"""
4h_Simple_Trend_Momentum_v1
Hypothesis: On 4h timeframe, use a simple 20-period EMA as trend filter and 14-period RSI for momentum entries. Go long when price > EMA20 and RSI crosses above 50; short when price < EMA20 and RSI crosses below 50. Add volume confirmation (>1.5x 20-period average) to filter weak moves. Target 20-40 trades/year per symbol to avoid fee drag. Works in bull via trend-following and in bear via short signals during downtrends.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA20 for trend filter
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    # RSI cross signals
    rsi_above_50 = rsi > 50
    rsi_below_50 = rsi < 50
    rsi_cross_up = np.zeros(n, dtype=bool)
    rsi_cross_down = np.zeros(n, dtype=bool)
    rsi_cross_up[1:] = (rsi_above_50[1:] & ~rsi_above_50[:-1])
    rsi_cross_down[1:] = (rsi_below_50[1:] & ~rsi_below_50[:-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for EMA20 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema20[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions with volume confirmation
        long_entry = (close[i] > ema20[i]) and rsi_cross_up[i] and vol_confirm[i]
        short_entry = (close[i] < ema20[i]) and rsi_cross_down[i] and vol_confirm[i]
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA20 or RSI crosses below 50
            if close[i] < ema20[i] or rsi_cross_down[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA20 or RSI crosses above 50
            if close[i] > ema20[i] or rsi_cross_up[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Simple_Trend_Momentum_v1"
timeframe = "4h"
leverage = 1.0