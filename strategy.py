#!/usr/bin/env python3
"""
4h_RSI_Extreme_Reversion_With_TrendFilter
Hypothesis: 4h RSI extreme reversion with 1d EMA50 trend filter and ATR stoploss.
- Long when RSI(14) < 30 (oversold) AND price > 1d EMA50 (uptrend filter)
- Short when RSI(14) > 70 (overbought) AND price < 1d EMA50 (downtrend filter)
- Uses ATR(14) trailing stop: exit long if price < highest_high_since_entry - 2.0 * ATR
- Uses ATR(14) trailing stop: exit short if price > lowest_low_since_entry + 2.0 * ATR
- Designed for low frequency (target 20-40 trades/year) to minimize fee drag
- Works in both bull and bear markets: trend filter avoids counter-trend trades,
  RSI extremes capture mean reversion in ranging markets and pullbacks in trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter (needs completed 1d candle)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend filter: 1 = uptrend (price > EMA50), -1 = downtrend (price < EMA50), 0 = invalid
    trend_filter = np.where(ema_50_1d_aligned > 0, 
                            np.where(close > ema_50_1d_aligned, 1, -1), 
                            0)
    
    # Calculate RSI(14) on primary timeframe (4h)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(loss_ma != 0, gain_ma / loss_ma, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # Track highest high since entry for longs, lowest low since entry for shorts
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Start after warmup (need 50 for 1d EMA, 14 for RSI and ATR)
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            # Update tracking variables
            if position == 1:
                highest_since_entry[i] = max(high[i], highest_since_entry[i-1]) if i > 0 else high[i]
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else low[i]
            elif position == -1:
                lowest_since_entry[i] = min(low[i], lowest_since_entry[i-1]) if i > 0 else low[i]
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else high[i]
            else:
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else high[i]
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else low[i]
            continue
        
        # Initialize tracking variables for current bar
        if i == start_idx:
            highest_since_entry[i] = high[i]
            lowest_since_entry[i] = low[i]
        else:
            highest_since_entry[i] = highest_since_entry[i-1]
            lowest_since_entry[i] = lowest_since_entry[i-1]
        
        # Update tracking variables based on position
        if position == 1:
            highest_since_entry[i] = max(high[i], highest_since_entry[i-1])
            lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == -1:
            lowest_since_entry[i] = min(low[i], lowest_since_entry[i-1])
            highest_since_entry[i] = highest_since_entry[i-1]
        
        # ATR-based stoploss conditions
        stop_long = False
        stop_short = False
        if position == 1 and highest_since_entry[i] > 0:
            stop_long = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
        elif position == -1 and lowest_since_entry[i] > 0:
            stop_short = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
        
        # RSI extreme reversion conditions with trend filter
        if position == 0:
            # Long: RSI < 30 (oversold) AND price > 1d EMA50 (uptrend)
            if rsi[i] < 30 and trend_filter[i] == 1:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
            # Short: RSI > 70 (overbought) AND price < 1d EMA50 (downtrend)
            elif rsi[i] > 70 and trend_filter[i] == -1:
                signals[i] = -0.25
                position = -1
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: RSI > 50 (mean reversion complete) OR ATR stoploss hit
            if rsi[i] > 50 or stop_long:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: RSI < 50 (mean reversion complete) OR ATR stoploss hit
            if rsi[i] < 50 or stop_short:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_RSI_Extreme_Reversion_With_TrendFilter"
timeframe = "4h"
leverage = 1.0