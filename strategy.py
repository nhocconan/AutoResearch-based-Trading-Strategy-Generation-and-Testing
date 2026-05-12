#!/usr/bin/env python3
# 4h_RSI_MeanReversion_Pullback
# Hypothesis: Mean reversion on 4h timeframe using RSI(14) oversold/overbought levels with price pullback to 20-period EMA.
# Works in both bull and bear markets by buying dips in uptrends and selling rallies in downtrends.
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.

name = "4h_RSI_MeanReversion_Pullback"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period EMA for pullback
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.2 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(rsi[i]) or np.isnan(ema20[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: RSI < 30 (oversold) + price near EMA20 + volume confirmation
            if rsi[i] < 30 and close[i] <= ema20[i] * 1.01 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought) + price near EMA20 + volume confirmation
            elif rsi[i] > 70 and close[i] >= ema20[i] * 0.99 and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 (mean reversion complete) or stoploss
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 (mean reversion complete) or stoploss
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals