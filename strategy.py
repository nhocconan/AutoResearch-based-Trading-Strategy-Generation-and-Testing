#!/usr/bin/env python3
"""
1d_KAMA_Direction_WeeklyTrend_1wRSI_Pullback
Hypothesis: Trade KAMA direction on daily timeframe with weekly RSI pullback confirmation. KAMA adapts to market efficiency, reducing whipsaw in sideways markets and capturing trends. Enter long when KAMA turns up and weekly RSI < 40 (pullback in uptrend), short when KAMA turns down and weekly RSI > 60 (pullback in downtrend). Uses volume > 1.5x 24-period average for confirmation. Designed for 1d timeframe to target 7-25 trades/year (30-100 total over 4 years). Uses weekly RSI for pullback to avoid chasing extended moves and align with higher timeframe trend.
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
    
    # Get weekly data for RSI
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly RSI(14)
    rsi_period = 14
    close_1w = df_1w['close'].values
    rsi_1w = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= rsi_period + 1:
        delta = np.diff(close_1w)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1w, np.nan)
        avg_loss = np.full_like(close_1w, np.nan)
        
        # First average
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close_1w)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1w = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # KAMA on daily timeframe (using close prices directly)
    # KAMA parameters
    fast_sc = 2 / (2 + 1)  # 2-period EMA
    slow_sc = 2 / (30 + 1) # 30-period EMA
    
    kama = np.full_like(close, np.nan)
    
    if len(close) >= 2:
        kama[0] = close[0]
        for i in range(1, len(close)):
            # Efficiency ratio
            if i >= 1:
                change = abs(close[i] - close[i-1])
                volatility = 0
                for j in range(1, i+1):
                    volatility += abs(close[j] - close[j-1])
                er = change / volatility if volatility != 0 else 0
                sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
                kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
            else:
                kama[i] = close[i]
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period)  # KAMA needs ~30 periods, vol MA needs 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: KAMA turning up + RSI pullback (<40) + volume
            if i > 0 and not np.isnan(kama[i-1]) and kama[i] > kama[i-1] and rsi_1w_aligned[i] < 40 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down + RSI pullback (>60) + volume
            elif i > 0 and not np.isnan(kama[i-1]) and kama[i] < kama[i-1] and rsi_1w_aligned[i] > 60 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down or RSI > 60 (overbought)
            if (i > 0 and not np.isnan(kama[i-1]) and kama[i] < kama[i-1]) or rsi_1w_aligned[i] > 60:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up or RSI < 40 (oversold)
            if (i > 0 and not np.isnan(kama[i-1]) and kama[i] > kama[i-1]) or rsi_1w_aligned[i] < 40:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_WeeklyTrend_1wRSI_Pullback"
timeframe = "1d"
leverage = 1.0