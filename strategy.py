#!/usr/bin/env python3
"""
6h_RSI_Momentum_FailSafe
Strategy: 6s RSI momentum with trend filter and volatility guard.
Long: RSI(14) > 55 + price > EMA(50) + ATR(14) > 0.3 * ATR(50)
Short: RSI(14) < 45 + price < EMA(50) + ATR(14) > 0.3 * ATR(50)
Exit: RSI returns to neutral zone (45-55) or volatility drops
Position size: 0.25
Designed to capture momentum bursts while avoiding chop and low volatility.
Timeframe: 6h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMAs
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_long = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_long[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Volatility filter: short-term ATR > 30% of long-term ATR
        vol_filter = atr[i] > (0.3 * atr_long[i])
        
        # Entry conditions
        if position == 0:
            # Long: RSI > 55 + price > EMA50 + volume + volatility
            if (rsi[i] > 55 and close[i] > ema_50[i] and 
                volume_filter and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 45 + price < EMA50 + volume + volatility
            elif (rsi[i] < 45 and close[i] < ema_50[i] and 
                  volume_filter and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI drops below 50 or volatility fails
            if rsi[i] < 50 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI rises above 50 or volatility fails
            if rsi[i] > 50 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_Momentum_FailSafe"
timeframe = "6h"
leverage = 1.0