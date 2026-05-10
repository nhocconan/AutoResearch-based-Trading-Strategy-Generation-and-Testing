#!/usr/bin/env python3
"""
6h_MultiFactor_Trend_Momentum
Hypothesis: Combine multiple factors (trend, momentum, volume) on 6h timeframe with weekly trend filter.
Uses 1w EMA50 for major trend direction, 6h RSI for momentum exhaustion, and 6h volume spike for confirmation.
Long when: 1) price above weekly EMA50 (uptrend), 2) RSI < 30 (oversold), 3) volume > 1.5x 20-period average.
Short when: 1) price below weekly EMA50 (downtrend), 2) RSI > 70 (overbought), 3) volume > 1.5x 20-period average.
Exit when RSI crosses 50 (mean reversion) or trend changes.
Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
Targets 50-120 trades over 4 years (12-30/year) to minimize fee drag.
"""

name = "6h_MultiFactor_Trend_Momentum"
timeframe = "6h"
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
    
    # 1w EMA50 for major trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 6h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Wilder's smoothing (equivalent to alpha = 1/14)
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        alpha = 1 / 14
        for i in range(14, n):
            avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
            avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.divide(avg_gain, avg_loss, out=np.full(n, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 6h volume SMA20 for volume confirmation
    vol_sma20 = np.full(n, np.nan)
    if n >= 20:
        vol_sma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # Need all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_sma20[i]
        
        if position == 0:
            # Long: Uptrend + oversold + volume confirmation
            if (close[i] > ema50_1w_aligned[i] and 
                rsi[i] < 30 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + overbought + volume confirmation
            elif (close[i] < ema50_1w_aligned[i] and 
                  rsi[i] > 70 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses above 50 (overbought) or trend changes
            if rsi[i] > 50 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses below 50 (oversold) or trend changes
            if rsi[i] < 50 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals