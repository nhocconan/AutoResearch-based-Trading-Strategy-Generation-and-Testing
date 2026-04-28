#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Pullback_Volume_Regime
Hypothesis: Uses 4h KAMA direction (trend filter) with RSI pullback (mean reversion) and volume confirmation to enter trades in the direction of trend. Designed for low trade frequency (15-40/year) by requiring KAMA trend alignment, RSI extreme pullback, and volume spike. Works in both bull and bear markets by following KAMA trend direction. Uses volume spike (2x 48-bar average) as confirmation filter to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on price
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >2x 48-period MA
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend filter
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # RSI pullback conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Volume confirmation
        vol_confirm = volume[i] > (2.0 * vol_ma_48[i])
        
        # Entry conditions: trend-aligned RSI pullback with volume
        long_entry = uptrend and rsi_oversold and vol_confirm
        short_entry = downtrend and rsi_overbought and vol_confirm
        
        # Exit conditions: opposite RSI extreme or trend reversal
        long_exit = rsi[i] > 70 or close[i] < kama[i]
        short_exit = rsi[i] < 30 or close[i] > kama[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Direction_RSI_Pullback_Volume_Regime"
timeframe = "4h"
leverage = 1.0