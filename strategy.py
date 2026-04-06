#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour KAMA Trend with Volume Confirmation and RSI Filter.
# Uses Kaufman's Adaptive Moving Average (KAMA) to identify trend direction.
# Enters long when price crosses above KAMA with RSI < 60 and volume > 1.5x average.
# Enters short when price crosses below KAMA with RSI > 40 and volume > 1.5x average.
# Works in both bull and bear markets by adapting to volatility and filtering weak signals.
# Target: 75-200 trades over 4 years (19-50/year).

name = "4h_kama_trend_volume_rsi_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA calculation (10-period ER, 2 and 30 for smoothing constants)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10)).values
    volatility = abs(close_s.diff(1)).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if data not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] < kama[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] > kama[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: price crosses above KAMA with RSI < 60
                if (close[i] > kama[i] and close[i-1] <= kama[i-1] and rsi[i] < 60):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price crosses below KAMA with RSI > 40
                elif (close[i] < kama[i] and close[i-1] >= kama[i-1] and rsi[i] > 40):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals