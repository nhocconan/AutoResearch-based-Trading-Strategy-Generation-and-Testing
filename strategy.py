#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_Exit
# Hypothesis: 4h KAMA trend with RSI-based exits and volume confirmation.
# KAMA adapts to market noise, reducing whipsaw in sideways markets.
# RSI exits capture mean reversion within trends. Volume filters ensure conviction.
# Designed for low trade frequency (20-50/year) with strong trend capture.

name = "4h_KAMA_Trend_RSI_Exit"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA trend indicator ---
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # needs correction
    # Correct volatility calculation: sum of absolute changes over window
    volatility = pd.Series(close).rolling(window=10, min_periods=10).apply(
        lambda x: np.sum(np.abs(np.diff(x))), raw=True
    ).values
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- RSI for mean reversion exits ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Volume confirmation (1.5x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: ensure we have enough data for indicators
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above KAMA (uptrend) with volume surge
            if close[i] > kama[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) with volume surge
            elif close[i] < kama[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: RSI overbought (mean reversion signal)
                if rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI oversold (mean reversion signal)
                if rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals