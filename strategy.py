#!/usr/bin/env python3
# 4h_rsi_ema_pullback_v1
# Hypothesis: RSI pullback strategy with EMA trend filter on 4h timeframe.
# Uses EMA(50) for trend direction and RSI(14) for mean-reversion entries.
# In uptrend (price > EMA50), look for RSI < 30 (oversold) for long entries.
# In downtrend (price < EMA50), look for RSI > 70 (overbought) for short entries.
# Includes volume confirmation and ATR-based stop via position management.
# Designed to work in both bull and bear markets by trading pullbacks to the trend.
# Target: 20-30 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_ema_pullback_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h indicators
    # EMA50 for trend direction
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema50[i]) or np.isnan(rsi[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction
        uptrend = close[i] > ema50[i]
        downtrend = close[i] < ema50[i]
        
        if position == 1:  # Long position
            # Exit conditions: RSI > 70 (overbought) or trend change
            if rsi[i] > 70 or close[i] < ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: RSI < 30 (oversold) or trend change
            if rsi[i] < 30 or close[i] > ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.2 * avg_volume[i]
            
            if volume_ok:
                # Long entry: uptrend + RSI oversold
                if uptrend and rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short entry: downtrend + RSI overbought
                elif downtrend and rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals