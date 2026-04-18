#!/usr/bin/env python3
"""
12h_Adaptive_RSI_Confluence - 12H strategy using adaptive RSI with volume and trend filters.
Long: RSI < 30 (oversold) + volume > 1.5x average + price above 20-period SMA
Short: RSI > 70 (overbought) + volume > 1.5x average + price below 20-period SMA
Exit: Opposite RSI extreme or trend reversal
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in both bull and bear markets by capturing mean reversion within trends.
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
    
    # Get daily data for multi-timeframe context
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period SMA for trend filter
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for SMA20 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_20[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to SMA20
        above_sma = close[i] > sma_20[i]
        below_sma = close[i] < sma_20[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # RSI conditions
        oversold = rsi[i] < 30
        overbought = rsi[i] > 70
        
        if position == 0:
            # Long: oversold + volume + price above SMA20
            if oversold and vol_confirm and above_sma:
                signals[i] = 0.25
                position = 1
            # Short: overbought + volume + price below SMA20
            elif overbought and vol_confirm and below_sma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: overbought condition or price below SMA20
            if overbought or not above_sma:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: oversold condition or price above SMA20
            if oversold or not below_sma:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Adaptive_RSI_Confluence"
timeframe = "12h"
leverage = 1.0