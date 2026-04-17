#!/usr/bin/env python3
"""
Hypothesis: 6h RSI(14) reversal with 1w trend filter and volume confirmation
- Long: RSI(14) < 30 (oversold) + price > 1w EMA50 + volume > 1.5x 20-period avg
- Short: RSI(14) > 70 (overbought) + price < 1w EMA50 + volume > 1.5x 20-period avg
- Exit: RSI crosses back to neutral zone (40-60)
- Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
- Target: 15-30 trades/year per symbol to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 6h timeframe
    ema50_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need weekly EMA50 (50) + RSI (14) + volume MA20 (20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema50_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price > weekly EMA50 + volume
            if (rsi[i] < 30 and close[i] > ema50_6h[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + price < weekly EMA50 + volume
            elif (rsi[i] > 70 and close[i] < ema50_6h[i] and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 40 (leaving oversold zone)
            if rsi[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 60 (leaving overbought zone)
            if rsi[i] < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI14_Reversal_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0