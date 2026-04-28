#!/usr/bin/env python3
"""
4h_RSI_Divergence_Volume_Trend
Hypothesis: RSI divergence with volume confirmation and daily trend filter works in both bull and bear markets. RSI divergence captures momentum exhaustion, volume confirms institutional participation, and daily trend filter reduces whipsaw. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate RSI peaks and troughs for divergence
    rsi_peak = np.zeros_like(rsi)
    rsi_trough = np.zeros_like(rsi)
    price_peak = np.zeros_like(close)
    price_trough = np.zeros_like(close)
    
    # Find local peaks and troughs (3-bar window)
    for i in range(2, n-2):
        # RSI peak: higher than neighbors
        if rsi[i] > rsi[i-1] and rsi[i] > rsi[i-2] and rsi[i] > rsi[i+1] and rsi[i] > rsi[i+2]:
            rsi_peak[i] = rsi[i]
            price_peak[i] = close[i]
        # RSI trough: lower than neighbors
        if rsi[i] < rsi[i-1] and rsi[i] < rsi[i-2] and rsi[i] < rsi[i+1] and rsi[i] < rsi[i+2]:
            rsi_trough[i] = rsi[i]
            price_trough[i] = close[i]
    
    # Forward fill peaks and troughs for comparison
    rsi_peak_series = pd.Series(rsi_peak).replace(0, np.nan).ffill().fillna(0).values
    rsi_trough_series = pd.Series(rsi_trough).replace(0, np.nan).ffill().fillna(0).values
    price_peak_series = pd.Series(price_peak).replace(0, np.nan).ffill().fillna(0).values
    price_trough_series = pd.Series(price_trough).replace(0, np.nan).ffill().fillna(0).values
    
    # Bullish divergence: price makes lower low, RSI makes higher low
    bullish_div = (price_trough_series < np.roll(price_trough_series, 1)) & (rsi_trough_series > np.roll(rsi_trough_series, 1))
    # Bearish divergence: price makes higher high, RSI makes lower high
    bearish_div = (price_peak_series > np.roll(price_peak_series, 1)) & (rsi_peak_series < np.roll(rsi_peak_series, 1))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = bullish_div[i] and volume_confirm[i] and uptrend
        short_entry = bearish_div[i] and volume_confirm[i] and downtrend
        
        # Exit on opposite signal
        long_exit = bearish_div[i] and volume_confirm[i]
        short_exit = bullish_div[i] and volume_confirm[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_Divergence_Volume_Trend"
timeframe = "4h"
leverage = 1.0