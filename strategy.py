#!/usr/bin/env python3
"""
4h_1d_Momentum_Divergence_v1
Hypothesis: Use daily RSI divergence with 4h price action and volume confirmation.
Enter long when price makes higher low but RSI makes lower low (bullish divergence) on daily,
    with 4h close above EMA50 and volume > 1.5x average.
Enter short when price makes lower high but RSI makes higher high (bearish divergence) on daily,
    with 4h close below EMA50 and volume > 1.5x average.
Exit when divergence fails or price crosses EMA50.
Designed to capture reversals in both bull and bear markets with low trade frequency.
Target: 50-120 total trades over 4 years (12-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Momentum_Divergence_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY RSI FOR DIVERGENCE ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_1d_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 4HOUR TREND FILTER ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_4h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Track daily pivots for divergence detection
    rsi_peaks = np.full(len(rsi_1d), np.nan)
    rsi_troughs = np.full(len(rsi_1d), np.nan)
    price_peaks = np.full(len(close_1d), np.nan)
    price_troughs = np.full(len(close_1d), np.nan)
    
    # Find peaks and troughs in daily RSI and price
    for i in range(2, len(rsi_1d)-2):
        # RSI peaks
        if rsi_1d[i] > rsi_1d[i-1] and rsi_1d[i] > rsi_1d[i-2] and \
           rsi_1d[i] > rsi_1d[i+1] and rsi_1d[i] > rsi_1d[i+2]:
            rsi_peaks[i] = rsi_1d[i]
        # RSI troughs
        if rsi_1d[i] < rsi_1d[i-1] and rsi_1d[i] < rsi_1d[i-2] and \
           rsi_1d[i] < rsi_1d[i+1] and rsi_1d[i] < rsi_1d[i+2]:
            rsi_troughs[i] = rsi_1d[i]
        # Price peaks
        if close_1d[i] > close_1d[i-1] and close_1d[i] > close_1d[i-2] and \
           close_1d[i] > close_1d[i+1] and close_1d[i] > close_1d[i+2]:
            price_peaks[i] = close_1d[i]
        # Price troughs
        if close_1d[i] < close_1d[i-1] and close_1d[i] < close_1d[i-2] and \
           close_1d[i] < close_1d[i+1] and close_1d[i] < close_1d[i+2]:
            price_troughs[i] = close_1d[i]
    
    # Align pivot points to 4h
    rsi_peaks_4h = align_htf_to_ltf(prices, df_1d, rsi_peaks)
    rsi_troughs_4h = align_htf_to_ltf(prices, df_1d, rsi_troughs)
    price_peaks_4h = align_htf_to_ltf(prices, df_1d, price_peaks)
    price_troughs_4h = align_htf_to_ltf(prices, df_1d, price_troughs)
    
    # Track last divergence signals
    last_bull_div = np.full(n, False)
    last_bear_div = np.full(n, False)
    
    # Check for bullish divergence: price higher low, RSI lower low
    for i in range(2, len(price_troughs_4h)):
        if not np.isnan(price_troughs_4h[i]) and not np.isnan(price_troughs_4h[i-2]):
            if (price_troughs_4h[i] > price_troughs_4h[i-2] and  # Higher low in price
                not np.isnan(rsi_troughs_4h[i]) and not np.isnan(rsi_troughs_4h[i-2]) and
                rsi_troughs_4h[i] < rsi_troughs_4h[i-2]):  # Lower low in RSI
                last_bull_div[i] = True
    
    # Check for bearish divergence: price lower high, RSI higher high
    for i in range(2, len(price_peaks_4h)):
        if not np.isnan(price_peaks_4h[i]) and not np.isnan(price_peaks_4h[i-2]):
            if (price_peaks_4h[i] < price_peaks_4h[i-2] and  # Lower high in price
                not np.isnan(rsi_peaks_4h[i]) and not np.isnan(rsi_peaks_4h[i-2]) and
                rsi_peaks_4h[i] > rsi_peaks_4h[i-2]):  # Higher high in RSI
                last_bear_div[i] = True
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(rsi_1d_4h[i]) or np.isnan(ema50_4h_4h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        trend_up = close[i] > ema50_4h_4h[i]
        trend_down = close[i] < ema50_4h_4h[i]
        
        # Long: bullish divergence with uptrend filter and volume surge
        long_signal = (last_bull_div[i] and 
                      trend_up and 
                      vol_ratio[i] > 1.5)
        
        # Short: bearish divergence with downtrend filter and volume surge
        short_signal = (last_bear_div[i] and 
                       trend_down and 
                       vol_ratio[i] > 1.5)
        
        # Exit: divergence fails or trend crosses EMA50
        exit_long = (position == 1 and 
                    (not trend_up or not last_bull_div[i]))
        exit_short = (position == -1 and 
                     (not trend_down or not last_bear_div[i]))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals