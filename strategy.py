#!/usr/bin/env python3
"""
6h_RSI_Divergence_1dTrend_VolumeConfirm_v1
Hypothesis: RSI divergence on 6h with 1d EMA50 trend filter and volume confirmation captures exhaustion moves in both bull and bear markets. Divergence signals weakening momentum; volume spike confirms validity. Trend filter ensures alignment with higher timeframe direction. Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection on 6h (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(rsi[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (EMA50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= 5:  # Need at least 5 bars to check divergence
            # Look back up to 10 bars for a lower low in price with higher low in RSI
            for lookback in range(2, min(11, i+1)):
                if low[i] < low[i-lookback] and rsi[i] > rsi[i-lookback]:
                    # Ensure intermediate lows are higher (valid divergence)
                    valid = True
                    for j in range(1, lookback):
                        if low[i-j] <= low[i-lookback] or rsi[i-j] >= rsi[i-lookback]:
                            valid = False
                            break
                    if valid:
                        bullish_div = True
                        break
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= 5:
            for lookback in range(2, min(11, i+1)):
                if high[i] > high[i-lookback] and rsi[i] < rsi[i-lookback]:
                    valid = True
                    for j in range(1, lookback):
                        if high[i-j] >= high[i-lookback] or rsi[i-j] <= rsi[i-lookback]:
                            valid = False
                            break
                    if valid:
                        bearish_div = True
                        break
        
        # Long logic: bullish divergence + volume spike + in uptrend or ranging (not strong downtrend)
        if bullish_div and volume_spike[i] and (uptrend or not downtrend):
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: bearish divergence + volume spike + in downtrend or ranging (not strong uptrend)
        elif bearish_div and volume_spike[i] and (downtrend or not uptrend):
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: opposite divergence or trend reversal
        elif position == 1 and (bearish_div or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bullish_div or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_Divergence_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0