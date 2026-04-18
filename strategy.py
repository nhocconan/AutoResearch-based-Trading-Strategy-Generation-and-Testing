#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Extreme_Volume
Hypothesis: On daily timeframe, use KAMA to identify trend direction, RSI extremes for entry timing, and volume surge for confirmation.
KAMA adapts to market noise, reducing false signals in chop. RSI < 30 or > 70 with volume spike indicates exhaustion reversal.
Trades only in direction of KAMA trend to avoid counter-trend whipsaws. Designed for low frequency (<20 trades/year) with high edge.
Works in bull (riding momentum) and bear (catching bounces/retracements) by following adaptive trend.
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
    
    # Get weekly data for trend filter (more robust than daily)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly KAMA trend
    close_1w = df_1w['close'].values
    kama_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 2:
        # Efficiency ratio
        change = np.abs(np.diff(close_1w))
        volatility = np.sum(np.abs(np.diff(close_1w)), axis=0) if len(close_1w) > 1 else 0
        # Simplified: use close price change over 10 periods
        er = np.full(len(close_1w), np.nan)
        for i in range(10, len(close_1w)):
            change_val = np.abs(close_1w[i] - close_1w[i-10])
            volatility_val = np.sum(np.abs(np.diff(close_1w[i-10:i+1])))
            er[i] = change_val / volatility_val if volatility_val > 0 else 0
            # Smoothing constants
            sc = (er[i] * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
            if i == 10:
                kama_1w[i] = close_1w[i]
            else:
                kama_1w[i] = kama_1w[i-1] + sc * (close_1w[i] - kama_1w[i-1])
    
    # Get daily data for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    rsi_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 15:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(len(close_1d), np.nan)
        avg_loss = np.full(len(close_1d), np.nan)
        avg_gain[14] = np.mean(gain[1:15])
        avg_loss[14] = np.mean(loss[1:15])
        for i in range(15, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
            rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi_1d[i] = 100 - (100 / (1 + rs))
    
    # Calculate daily volume average
    vol_ma_1d = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        vol_ma_1d[i] = np.mean(volume[i-20:i]) if i < len(volume) else np.nan
    
    # Align to daily timeframe (since we're using 1d primary)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    rsi_1d_aligned = rsi_1d  # already daily
    vol_ma_1d_aligned = vol_ma_1d  # already daily
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for KAMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge: current volume > 2.5 x 20-day average
        vol_surge = volume[i] > (vol_ma_1d_aligned[i] * 2.5) if not np.isnan(vol_ma_1d_aligned[i]) else False
        
        if position == 0:
            # Long: KAMA uptrend + RSI oversold + volume surge
            if (close[i] > kama_1w_aligned[i] and 
                rsi_1d_aligned[i] < 30 and vol_surge):
                signals[i] = 0.25
                position = 1
            # Short: KAMA downtrend + RSI overbought + volume surge
            elif (close[i] < kama_1w_aligned[i] and 
                  rsi_1d_aligned[i] > 70 and vol_surge):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or trend turns down
            if (rsi_1d_aligned[i] > 70 or close[i] < kama_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or trend turns up
            if (rsi_1d_aligned[i] < 30 or close[i] > kama_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_Extreme_Volume"
timeframe = "1d"
leverage = 1.0