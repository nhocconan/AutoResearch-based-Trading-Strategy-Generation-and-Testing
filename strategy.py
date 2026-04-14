#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day RSI extremes with 1-week volume-weighted average price (VWAP) as dynamic support/resistance.
# Long when RSI < 30 (oversold) AND price > weekly VWAP (bullish bias).
# Short when RSI > 70 (overbought) AND price < weekly VWAP (bearish bias).
# Exit when RSI returns to neutral range (40-60).
# Uses RSI for mean-reversion signals in extreme conditions and weekly VWAP for trend bias.
# Designed to work in both bull and bear markets by fading extremes only when aligned with higher timeframe trend.
# Target: 20-35 trades/year per symbol (80-140 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for RSI and 1w data for VWAP
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 14:  # Need enough for RSI
        return np.zeros(n)
    
    # Calculate RSI (14) on daily timeframe
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate VWAP on weekly timeframe
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    vwap_numerator = np.cumsum(typical_price_1w * volume_1w)
    vwap_denominator = np.cumsum(volume_1w)
    vwap = vwap_numerator / (vwap_denominator + 1e-10)  # Avoid division by zero
    
    # Align indicators to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for RSI calculation
    start = 14
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(vwap_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for RSI extremes with VWAP filter
            # Long: RSI < 30 (oversold) AND price > weekly VWAP (bullish bias)
            if (rsi_aligned[i] < 30 and 
                close[i] > vwap_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) AND price < weekly VWAP (bearish bias)
            elif (rsi_aligned[i] > 70 and 
                  close[i] < vwap_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral range (40-60)
            if (rsi_aligned[i] >= 40 and 
                rsi_aligned[i] <= 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral range (40-60)
            if (rsi_aligned[i] >= 40 and 
                rsi_aligned[i] <= 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dRSI_1wVWAP_ExtremeReversion_v1"
timeframe = "4h"
leverage = 1.0