#!/usr/bin/env python3
"""
1h RSI(14) + 4h EMA(50) + Volume Spike + Session Filter
Hypothesis: RSI mean reversion on 1h with 4h EMA trend filter and volume confirmation works in both bull and bear markets. Session filter (08-20 UTC) reduces noise. Target 60-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi14_ema50_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi = np.zeros(n)
    
    # Wilder's smoothing
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100 if avg_gain[i] > 0 else 0
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 49) / 51
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h Volume spike filter (20-period average)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(50, 20)  # warmup for EMA and volume
    
    for i in range(start, n):
        # Skip if data not ready
        if np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session check
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume spike: current volume > 1.5 * 20-period average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long
            # Exit: RSI > 60 (overbought) or close below 4h EMA
            if rsi[i] > 60 or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short
            # Exit: RSI < 40 (oversold) or close above 4h EMA
            if rsi[i] < 40 or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extreme + volume spike + session + 4h EMA filter
            if in_session and volume_spike:
                # Long: RSI < 30 (oversold) and price above 4h EMA (uptrend)
                if rsi[i] < 30 and close[i] > ema_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: RSI > 70 (overbought) and price below 4h EMA (downtrend)
                elif rsi[i] > 70 and close[i] < ema_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals