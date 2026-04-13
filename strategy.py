#!/usr/bin/env python3
"""
6h_1d_RSI_Divergence_Volume
Hypothesis: Uses bearish/bullish RSI divergences on 1d timeframe combined with volume exhaustion signals on 6h.
In both bull and bear markets, price often makes higher highs/lows while RSI fails to confirm, signaling weakening momentum.
Volume exhaustion (decreasing volume on price moves) confirms the divergence.
Target: 15-30 trades/year on 6h (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def find_divergence(high, low, rsi, lookback=5):
    """Find bearish and bullish divergences.
    Bearish: price makes higher high, RSI makes lower high
    Bullish: price makes lower low, RSI makes higher low
    Returns arrays of same length with 1 for bearish, -1 for bullish, 0 otherwise.
    """
    n = len(high)
    bearish = np.zeros(n)
    bullish = np.zeros(n)
    
    for i in range(lookback, n):
        # Check for bearish divergence: price HH, RSI LH
        price_hh = high[i] == np.max(high[i-lookback:i+1])
        rsi_lh = rsi[i] < np.max(rsi[i-lookback:i])  # RSI lower than recent high
        
        # Check for bullish divergence: price LL, RSI HL
        price_ll = low[i] == np.min(low[i-lookback:i+1])
        rsi_hl = rsi[i] > np.min(rsi[i-lookback:i])  # RSI higher than recent low
        
        if price_hh and rsi_lh:
            bearish[i] = 1
        elif price_ll and rsi_hl:
            bullish[i] = -1
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI and divergence calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate RSI on daily closes
    rsi_1d = calculate_rsi(close_1d, period=14)
    
    # Find divergences on daily
    bearish_div, bullish_div = find_divergence(high_1d, low_1d, rsi_1d, lookback=10)
    
    # Volume exhaustion: current volume < 0.7 * 20-period average (weakening momentum)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_exhaustion = volume_1d < (vol_ma_20 * 0.7)
    
    # Combine signals: divergence + volume exhaustion
    bearish_signal = bearish_div & volume_exhaustion
    bullish_signal = bullish_div & volume_exhaustion
    
    # Align all data to 6h timeframe
    bearish_signal_aligned = align_htf_to_ltf(prices, df_1d, bearish_signal.astype(float))
    bullish_signal_aligned = align_htf_to_ltf(prices, df_1d, bullish_signal.astype(float))
    
    # Volume confirmation on 6t: current volume > 1.2 * 20-period average (participation)
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_participation = volume > (vol_ma_20_6h * 1.2)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(bearish_signal_aligned[i]) or np.isnan(bullish_signal_aligned[i]) or 
            np.isnan(volume_participation[i])):
            signals[i] = 0.0
            continue
        
        # Enter short on bearish divergence with volume exhaustion and participation
        short_condition = bearish_signal_aligned[i] > 0.5 and volume_participation[i]
        
        # Enter long on bullish divergence with volume exhaustion and participation
        long_condition = bullish_signal_aligned[i] < -0.5 and volume_participation[i]
        
        if short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_condition and position == -1:
            signals[i] = -position_size
        elif long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_condition and position == 1:
            signals[i] = position_size
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_RSI_Divergence_Volume"
timeframe = "6h"
leverage = 1.0