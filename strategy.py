#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 200-day EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200-day EMA on daily closes
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 200-day EMA to 4h timeframe (wait for daily close)
    ema_200_4h = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # 4-period RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_200_4h[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 200-day EMA
        uptrend = close[i] > ema_200_4h[i]
        downtrend = close[i] < ema_200_4h[i]
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        if position == 0:
            # Long: uptrend + RSI oversold bounce + volume
            if uptrend and rsi[i] < 30 and rsi[i] > rsi[i-1] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + RSI overbought rejection + volume
            elif downtrend and rsi[i] > 70 and rsi[i] < rsi[i-1] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend change
            if rsi[i] > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or trend change
            if rsi[i] < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA200_RSI4_VolumeFilter"
timeframe = "4h"
leverage = 1.0