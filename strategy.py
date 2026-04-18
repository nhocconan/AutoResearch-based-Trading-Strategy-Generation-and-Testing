#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_1d_RSI_Filter_v1
Hypothesis: Use 12h KAMA to determine trend direction and 1d RSI for momentum confirmation.
Go long when KAMA is rising and 1d RSI > 50, short when KAMA is falling and 1d RSI < 50.
Requires volume > 1.3x 20-period average for confirmation.
Target: 15-30 trades/year by using trend-following with momentum filter to reduce noise.
Works in bull markets via trend following and in bear via short signals.
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
    
    # Get 12h data for KAMA
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h KAMA(10)
    kama_period = 10
    kama_12h = np.full_like(close_12h, np.nan)
    
    if len(close_12h) >= kama_period:
        # Efficiency ratio
        change = np.abs(np.diff(close_12h, kama_period))
        volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)
        # Handle edge cases
        er = np.zeros_like(close_12h)
        for i in range(kama_period, len(close_12h)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        
        # First KAMA value
        kama_12h[kama_period] = close_12h[kama_period]
        
        # Subsequent values
        for i in range(kama_period + 1, len(close_12h)):
            kama_12h[i] = kama_12h[i-1] + sc[i] * (close_12h[i] - kama_12h[i-1])
    
    # Align KAMA to 12h timeframe (no shift needed as we're already on 12h)
    kama_12h_aligned = kama_12h  # Already on 12h timeframe
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d RSI(14)
    rsi_period = 14
    rsi_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= rsi_period + 1:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1d, np.nan)
        avg_loss = np.full_like(close_1d, np.nan)
        
        # First average
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kama_period, rsi_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: rising if current > previous
        kama_rising = kama_12h_aligned[i] > kama_12h_aligned[i-1] if i > 0 else False
        kama_falling = kama_12h_aligned[i] < kama_12h_aligned[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: KAMA rising AND RSI > 50 AND volume confirmation
            if kama_rising and rsi_1d_aligned[i] > 50 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND RSI < 50 AND volume confirmation
            elif kama_falling and rsi_1d_aligned[i] < 50 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA falling OR RSI < 40
            if kama_falling or rsi_1d_aligned[i] < 40:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising OR RSI > 60
            if kama_rising or rsi_1d_aligned[i] > 60:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_With_1d_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0