#!/usr/bin/env python3
# 4H_RSI_PULLBACK_TO_50_EMA_WITH_VOLUME_CONFIRMATION
# Hypothesis: RSI(14) > 70 followed by pullback to 50 EMA on 4H with volume confirmation captures mean reversion in overbought conditions.
# Works in bull markets (buy pullbacks in uptrends) and bear markets (sell rallies in downtrends) by fading extreme RSI moves.
# Target: 20-30 trades/year on 4H timeframe to avoid overtrading.

name = "4H_RSI_PULLBACK_TO_50_EMA_WITH_VOLUME_CONFIRMATION"
timeframe = "4h"
leverage = 1.0

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
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 50 EMA
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after RSI warmup
        # Exit conditions
        if position == 1:
            if close[i] >= ema50[i] or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] <= ema50[i] or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # LONG: RSI > 70 (overbought) then pullback to EMA50 with volume
            if (rsi[i-1] > 70 and 
                close[i] <= ema50[i] * 1.005 and  # Allow small buffer
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 30 (oversold) then rally to EMA50 with volume
            elif (rsi[i-1] < 30 and 
                  close[i] >= ema50[i] * 0.995 and  # Allow small buffer
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals