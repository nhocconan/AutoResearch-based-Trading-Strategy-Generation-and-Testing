#!/usr/bin/env python3
# 6h_1d_Momentum_Divergence_With_Volume
# Hypothesis: Combine 1d momentum divergence (RSI) with 6s price action and volume confirmation.
# In bull markets: Buy when price makes higher low but RSI makes lower low (bullish divergence) on 1d, confirmed by 6s breakout.
# In bear markets: Sell when price makes lower high but RSI makes higher high (bearish divergence) on 1d, confirmed by 6s breakdown.
# Uses volume surge to filter false signals. Targets 50-150 total trades over 4 years.

name = "6h_1d_Momentum_Divergence_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def rsi(close, period=14):
    """Calculate RSI with proper Wilder's smoothing."""
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's smoothing: alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values  # Fill neutral for early periods

def find_divergences(high, low, rsi_vals, lookback=5):
    """
    Find bullish and bearish divergences.
    Bullish: price makes lower low, RSI makes higher low
    Bearish: price makes higher high, RSI makes lower high
    Returns arrays with 1 for bullish div, -1 for bearish div, 0 otherwise
    """
    n = len(high)
    div_signal = np.zeros(n)
    
    for i in range(lookback, n):
        # Check for bullish divergence: lower low in price, higher low in RSI
        price_low_idx = np.argmin(low[i-lookback:i+1]) + i - lookback
        rsi_low_idx = np.argmin(rsi_vals[i-lookback:i+1]) + i - lookback
        
        if (low[price_low_idx] < low[i-lookback:i].min() and 
            rsi_vals[rsi_low_idx] > rsi_vals[i-lookback:i].min()):
            div_signal[i] = 1  # Bullish divergence
            
        # Check for bearish divergence: higher high in price, lower high in RSI
        price_high_idx = np.argmax(high[i-lookback:i+1]) + i - lookback
        rsi_high_idx = np.argmax(rsi_vals[i-lookback:i+1]) + i - lookback
        
        if (high[price_high_idx] > high[i-lookback:i].max() and 
            rsi_vals[rsi_high_idx] < rsi_vals[i-lookback:i].max()):
            div_signal[i] = -1  # Bearish divergence
            
    return div_signal

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for RSI
        return np.zeros(n)
    
    # Calculate 1d RSI for divergence detection
    close_1d = df_1d['close'].values
    rsi_1d = rsi(close_1d, 14)
    
    # Find divergences on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    div_signal_1d = find_divergences(high_1d, low_1d, rsi_1d, lookback=10)
    
    # Align divergence signal to 6h timeframe
    div_aligned = align_htf_to_ltf(prices, df_1d, div_signal_1d)
    
    # 6h breakout levels: 20-period high/low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(div_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish divergence on 1d + break above 20-period high + volume surge
            if (div_aligned[i] == 1 and 
                close[i] > high_20[i] * 1.002 and 
                volume[i] > 1.8 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence on 1d + break below 20-period low + volume surge
            elif (div_aligned[i] == -1 and 
                  close[i] < low_20[i] * 0.998 and 
                  volume[i] > 1.8 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish divergence OR price breaks below 20-period low
            if (div_aligned[i] == -1 or close[i] < low_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish divergence OR price breaks above 20-period high
            if (div_aligned[i] == 1 or close[i] > high_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals