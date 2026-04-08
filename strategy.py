#!/usr/bin/env python3

# 4h_rsi_pullback_trend_v1
# Hypothesis: RSI pullback strategy in trending markets. Uses daily EMA200 for trend filter,
# RSI(14) for pullback entries (RSI<35 in uptrend, RSI>65 in downtrend), and volume confirmation.
# Works in both bull and bear markets by trading pullbacks within the dominant trend.
# Target: 25-35 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_trend_v1"
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
    
    # Daily trend filter (1d EMA200) - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily data
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h indicators
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Pad first element
    rsi = np.concatenate([[50.0], rsi])
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(avg_volume[i]) or np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema200_1d_aligned[i]
        daily_downtrend = close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend reversal
            if rsi[i] > 70 or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend reversal
            if rsi[i] < 30 or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            if volume_ok:
                # Long entry: RSI pullback in uptrend
                if daily_uptrend and rsi[i] < 35:
                    position = 1
                    signals[i] = 0.25
                # Short entry: RSI pullback in downtrend
                elif daily_downtrend and rsi[i] > 65:
                    position = -1
                    signals[i] = -0.25
    
    return signals