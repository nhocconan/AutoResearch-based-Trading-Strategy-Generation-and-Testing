#!/usr/bin/env python3
name = "1d_RSI_Candlestick_Pullback"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan
    
    # RSI moving average for trend filter
    rsi_sma = np.full(n, np.nan)
    for i in range(23, n):  # 14 + 9
        rsi_sma[i] = np.nanmean(rsi[i-8:i+1])
    
    # Bullish engulfing pattern
    bullish_engulf = np.zeros(n, dtype=bool)
    for i in range(1, n):
        bullish_engulf[i] = (close[i] > open_price[i-1]) and (open_price[i] < close[i-1]) and (close[i-1] < open_price[i-1])
    
    # Bearish engulfing pattern
    bearish_engulf = np.zeros(n, dtype=bool)
    for i in range(1, n):
        bearish_engulf[i] = (close[i] < open_price[i-1]) and (open_price[i] > close[i-1]) and (close[i-1] > open_price[i-1])
    
    # Volume filter: 20-day average
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-19:i+1])
    volume_ok = volume > vol_ma20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    start_idx = 30  # Ensure enough data for RSI and volume
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(rsi_sma[i]) or np.isnan(volume_ok[i]):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Long: RSI < 30 (oversold) + bullish engulfing + volume confirmation
        if rsi[i] < 30 and bullish_engulf[i] and volume_ok[i]:
            signals[i] = position_size
            position = 1
        # Short: RSI > 70 (overbought) + bearish engulfing + volume confirmation
        elif rsi[i] > 70 and bearish_engulf[i] and volume_ok[i]:
            signals[i] = -position_size
            position = -1
        # Exit: RSI returns to neutral zone (40-60)
        elif position == 1 and rsi[i] > 50:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi[i] < 50:
            signals[i] = 0.0
            position = 0
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals