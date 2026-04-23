#!/usr/bin/env python3
"""
Hypothesis: Daily RSI with weekly VWAP filter and volume confirmation.
Long when RSI crosses above 30 (oversold bounce) + price above weekly VWAP + volume > 1.5x average.
Short when RSI crosses below 70 (overbought rejection) + price below weekly VWAP + volume > 1.5x average.
Exit when RSI crosses 50 (mean reversion) or price crosses weekly VWAP.
Designed for low trade frequency (~10-20/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-week data for VWAP filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly VWAP
    weekly_tp = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_vwap = (weekly_tp * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    weekly_vwap_values = weekly_vwap.values
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, weekly_vwap_values)
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(weekly_vwap_aligned[i]) or 
            np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_vs_vwap = close[i] > weekly_vwap_aligned[i]
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: RSI crosses above 30 + price above weekly VWAP + volume confirmation
            if (rsi[i] > 30 and rsi[i-1] <= 30 and 
                price_vs_vwap and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 70 + price below weekly VWAP + volume confirmation
            elif (rsi[i] < 70 and rsi[i-1] >= 70 and 
                  not price_vs_vwap and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI crosses above 50 or price below weekly VWAP
                if rsi[i] >= 50 or not price_vs_vwap:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI crosses below 50 or price above weekly VWAP
                if rsi[i] <= 50 or price_vs_vwap:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_RSI_WeeklyVWAP_VolumeFilter"
timeframe = "1d"
leverage = 1.0