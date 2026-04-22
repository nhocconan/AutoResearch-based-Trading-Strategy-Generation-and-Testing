#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA(14,2,30) direction with RSI(14) pullback and volume confirmation
# Uses Kaufman's Adaptive Moving Average to identify trend direction efficiently.
# Enters on pullbacks to KAMA during established trends with volume confirmation.
# Designed to work in both bull and bear markets by following trend with mean-reversion entries.
# Target: 20-30 trades/year per symbol (80-120 total) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA ( Kaufman's Adaptive Moving Average )
    # Parameters: length=14, fast=2, slow=30
    kama_length = 14
    fast_sc = 2
    slow_sc = 30
    
    # Calculate change and volatility
    change = np.abs(close - np.roll(close, kama_length))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    
    # Handle first kama_length elements
    change[:kama_length] = 0
    volatility[:kama_length] = 0
    
    # Calculate efficiency ratio and smoothing constant
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[kama_length] = close[kama_length]
    for i in range(kama_length + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price pulls back to KAMA in uptrend (price > KAMA) with RSI < 40 and volume confirmation
            if (close[i] > kama[i] and rsi[i] < 40 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price pulls back to KAMA in downtrend (price < KAMA) with RSI > 60 and volume confirmation
            elif (close[i] < kama[i] and rsi[i] > 60 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses KAMA or RSI reaches extreme
            if position == 1:
                if close[i] < kama[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Pullback_Volume_Session"
timeframe = "4h"
leverage = 1.0