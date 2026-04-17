#!/usr/bin/env python3
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
    
    # Get 1d data for 14-period RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily closes
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 1w data for 50-period SMA (trend filter)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_12h = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need RSI, SMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_12h[i]) or 
            np.isnan(sma50_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1w SMA50
        price_above_sma = close[i] > sma50_12h[i]
        price_below_sma = close[i] < sma50_12h[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND price above 1w SMA50 with volume
            if (rsi_12h[i] < 30 and price_above_sma and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) AND price below 1w SMA50 with volume
            elif (rsi_12h[i] > 70 and price_below_sma and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (neutral) OR price crosses below 1w SMA50
            if (rsi_12h[i] > 50) or (close[i] < sma50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 (neutral) OR price crosses above 1w SMA50
            if (rsi_12h[i] < 50) or (close[i] > sma50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI_OverboughtOversold_1wSMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0