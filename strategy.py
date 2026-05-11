#!/usr/bin/env python3
"""
1d_1w_RSI_Bollinger_MeanReversion
Hypothesis: Uses weekly RSI to detect extreme conditions (overbought/oversold) and daily Bollinger Band touches for entry. 
In bull markets: buy weekly oversold RSI (<30) when price touches lower BB. 
In bear markets: sell weekly overbought RSI (>70) when price touches upper BB. 
Requires volume confirmation to avoid false signals. Designed for low trade frequency (<20/year) via strict weekly RSI filter.
"""

name = "1d_1w_RSI_Bollinger_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower, sma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly RSI for Extreme Conditions ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    rsi_1w = calculate_rsi(df_1w['close'].values, period=14)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # --- Daily Bollinger Bands ---
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, period=20, std_dev=2)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: weekly oversold RSI (<30) + price touches lower BB + volume
            if (rsi_1w_aligned[i] < 30 and 
                low[i] <= bb_lower[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: weekly overbought RSI (>70) + price touches upper BB + volume
            elif (rsi_1w_aligned[i] > 70 and 
                  high[i] >= bb_upper[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: RSI returns to neutral zone or price reaches middle band
            if position == 1:
                # Exit long: RSI > 50 OR price touches middle/upper band
                if rsi_1w_aligned[i] > 50 or close[i] >= bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI < 50 OR price touches middle/lower band
                if rsi_1w_aligned[i] < 50 or close[i] <= bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals