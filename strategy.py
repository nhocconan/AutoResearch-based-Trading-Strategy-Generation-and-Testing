#!/usr/bin/env python3
"""
6h_Keltner_RSI_Reversal_1dTrend_Volume
Hypothesis: On 6h timeframe, price reversals from Keltner Channel extremes (2*ATR) 
combined with RSI overbought/oversold conditions, filtered by 1d EMA50 trend 
and volume > 1.5x average, provide edge in both bull and bear markets.
Keltner Channels adapt to volatility, making them effective across regimes.
RSI extremes signal exhaustion. Volume confirms institutional interest.
Target: 60-120 total trades over 4 years (~15-30/year) to balance opportunity and cost.
"""

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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA to 6h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # ATR for Keltner Channels (2*ATR from EMA20)
    atr_period = 20
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # EMA(20) for Keltner middle line
    ema20_period = 20
    ema20 = np.full(n, np.nan)
    if n >= ema20_period:
        ema20[ema20_period-1] = np.mean(close[:ema20_period])
        multiplier = 2 / (ema20_period + 1)
        for i in range(ema20_period, n):
            ema20[i] = (close[i] * multiplier) + (ema20[i-1] * (1 - multiplier))
    
    # Keltner Channels: upper = EMA20 + 2*ATR, lower = EMA20 - 2*ATR
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr
    
    # RSI(14) for overbought/oversold
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(rsi_period, n):
        avg_gain[i] = np.mean(gain[i-rsi_period+1:i+1])
        avg_loss[i] = np.mean(loss[i-rsi_period+1:i+1])
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    
    # Warmup: need all indicators
    start_idx = max(atr_period, ema20_period, rsi_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or
            np.isnan(keltner_upper[i]) or
            np.isnan(keltner_lower[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if uptrend and volume_confirmation:
            # In uptrend, look for pullbacks to lower Keltner with RSI oversold
            if price <= keltner_lower[i] and rsi[i] < 30:
                signals[i] = 0.25  # Long 25%
            else:
                signals[i] = 0.0
        elif downtrend and volume_confirmation:
            # In downtrend, look for bounces to upper Keltner with RSI overbought
            if price >= keltner_upper[i] and rsi[i] > 70:
                signals[i] = -0.25  # Short 25%
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Keltner_RSI_Reversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0