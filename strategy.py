#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h trend filter and volume confirmation.
Use 4h EMA20 as trend filter, 1h RSI(14) for mean reversion entries, and volume spike for confirmation.
Target 15-30 trades/year by requiring: trend alignment, RSI extreme, and volume > 1.5x average.
Works in bull/bear: trend filter prevents counter-trend trades, mean reversion captures reversals.
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
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend
    close_4h = df_4h['close'].values
    ema_20_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 20:
        ema_20_4h[19] = np.mean(close_4h[:20])
        for i in range(20, len(close_4h)):
            ema_20_4h[i] = (close_4h[i] * 2 + ema_20_4h[i-1] * 19) / 21
    
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    if len(close) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, len(close)):
            avg_gain[i] = (gain[i] + avg_gain[i-1] * 13) / 14
            avg_loss[i] = (loss[i] + avg_loss[i-1] * 13) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume average (20-period)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: require at least 1.5x average volume
        if vol_ma_20[i] <= 0 or volume[i] < vol_ma_20[i] * 1.5:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Uptrend (price > EMA20) + RSI oversold (<30)
            if close[i] > ema_20_4h_aligned[i] and rsi[i] < 30:
                position = 1
                signals[i] = position_size
            # Short: Downtrend (price < EMA20) + RSI overbought (>70)
            elif close[i] < ema_20_4h_aligned[i] and rsi[i] > 70:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI overbought (>70) or trend breaks
            if rsi[i] > 70 or close[i] < ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI oversold (<30) or trend breaks
            if rsi[i] < 30 or close[i] > ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_EMA20_RSI_Volume_Filter"
timeframe = "1h"
leverage = 1.0