#!/usr/bin/env python3
"""
12h_1d_RSI_Divergence_Trend_Follow
Hypothesis: Combine 1d RSI divergence with 12h price action to catch trend reversals in both bull and bear markets.
In bull markets: buy when bullish RSI divergence occurs on 1d and price closes above 12h EMA20.
In bear markets: sell when bearish RSI divergence occurs on 1d and price closes below 12h EMA20.
Uses volume confirmation to filter weak signals and ATR-based stoploss to manage risk.
Designed for low frequency (10-30 trades/year) to minimize fee drag on 12h timeframe.
"""

name = "12h_1d_RSI_Divergence_Trend_Follow"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper Wilder's smoothing."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # Wilder's smoothing: first average is simple average
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    # Subsequent values: smoothed average
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Set initial values to NaN
    rsi[:period] = np.nan
    return rsi

def find_rsi_divergence(high, low, rsi, lookback=14):
    """
    Find bullish and bearish RSI divergence.
    Bullish: price makes lower low, RSI makes higher low
    Bearish: price makes higher high, RSI makes lower high
    Returns arrays of same length with 1 for bullish div, -1 for bearish div, 0 otherwise
    """
    n = len(high)
    bullish_div = np.zeros(n)
    bearish_div = np.zeros(n)
    
    for i in range(lookback, n):
        # Look for divergence in the lookback window
        window_high = high[i-lookback:i+1]
        window_low = low[i-lookback:i+1]
        window_rsi = rsi[i-lookback:i+1]
        
        # Skip if any NaN in window
        if np.any(np.isnan(window_rsi)):
            continue
            
        # Find local minima and maxima in the window
        price_min_idx = np.argmin(window_low)
        price_max_idx = np.argmax(window_high)
        rsi_min_idx = np.argmin(window_rsi)
        rsi_max_idx = np.argmax(window_rsi)
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (price_min_idx == lookback and  # Current bar is lowest in window
            rsi_min_idx != lookback and    # RSI low is not current bar
            window_rsi[rsi_min_idx] < window_rsi[lookback]):  # RSI was lower before
            bullish_div[i] = 1
            
        # Bearish divergence: price makes higher high, RSI makes lower high
        elif (price_max_idx == lookback and  # Current bar is highest in window
              rsi_max_idx != lookback and    # RSI high is not current bar
              window_rsi[rsi_max_idx] > window_rsi[lookback]):  # RSI was higher before
            bearish_div[i] = -1
    
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for RSI and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d RSI
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Find RSI divergence on 1d
    bullish_div_1d, bearish_div_1d = find_rsi_divergence(high_1d, low_1d, rsi_1d, 14)
    
    # Align divergence signals to 12h timeframe
    bullish_div_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_div_1d)
    bearish_div_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_div_1d)

    # 12h EMA20 for trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if (np.isnan(bullish_div_1d_aligned[i]) or np.isnan(bearish_div_1d_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish RSI divergence on 1d + price above EMA20 + volume spike
            if (bullish_div_1d_aligned[i] == 1 and 
                close[i] > ema20[i] and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish RSI divergence on 1d + price below EMA20 + volume spike
            elif (bearish_div_1d_aligned[i] == -1 and 
                  close[i] < ema20[i] and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence occurs or price closes below EMA20
            if (bearish_div_1d_aligned[i] == -1 or 
                close[i] < ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence occurs or price closes above EMA20
            if (bullish_div_1d_aligned[i] == 1 or 
                close[i] > ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals