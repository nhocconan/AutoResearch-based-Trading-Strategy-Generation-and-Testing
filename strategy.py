#!/usr/bin/env python3
"""
4h_rsi_pullback_1w_trend_v1
Hypothesis: RSI pullback on 4h chart with weekly trend filter and volume confirmation.
Long: RSI(14) < 30 (oversold) with price above weekly EMA50 and volume above average.
Short: RSI(14) > 70 (overbought) with price below weekly EMA50 and volume above average.
Uses RSI for mean reversion entries, weekly EMA for trend filter, and volume for confirmation.
Designed for 20-40 trades/year on 4h timeframe with clear rules that work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_1w_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if data not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral or trend turns bearish
            if rsi[i] >= 50 or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral or trend turns bullish
            if rsi[i] <= 50 or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: RSI oversold with volume confirmation and bullish trend
            if rsi[i] < 30 and vol_confirmed and close[i] > ema50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: RSI overbought with volume confirmation and bearish trend
            elif rsi[i] > 70 and vol_confirmed and close[i] < ema50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals