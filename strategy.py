#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 14-day RSI with 1-day volume-weighted average price (VWAP) as dynamic support/resistance
# RSI < 30 and price > 1d VWAP = long (oversold bounce in uptrend)
# RSI > 70 and price < 1d VWAP = short (overbought rejection in downtrend)
# 1d VWAP provides institutional reference level; RSI avoids chasing extremes
# Works in bull (buy dips to VWAP) and bear (sell rallies to VWAP) markets
# Target: 50-150 total trades over 4 years (12-37/year)
name = "6h_RSI_1dVWAP"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # 14-period RSI on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(vwap_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price above 1d VWAP
            if (rsi[i] < 30 and close[i] > vwap_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) and price below 1d VWAP
            elif (rsi[i] > 70 and close[i] < vwap_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI > 50 (momentum fading) or breaks below VWAP
            if (rsi[i] > 50) or (close[i] < vwap_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI < 50 (momentum fading) or breaks above VWAP
            if (rsi[i] < 50) or (close[i] > vwap_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals