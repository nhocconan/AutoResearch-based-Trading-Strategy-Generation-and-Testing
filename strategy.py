#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_1d_RSI_Confirmation_v1
Hypothesis: Use daily KAMA for trend direction and 1d RSI for momentum confirmation on 12h timeframe.
Go long when KAMA slope is positive AND RSI > 50, short when KAMA slope is negative AND RSI < 50.
Uses volume confirmation (>1.2x 20-period average) and session filter (08-20 UTC).
Target: 20-30 trades/year by combining trend and momentum filters to reduce noise.
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
    volume = prices['volume'].values
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d KAMA(10,2,30)
    kama_period = 10
    fast_sc = 2
    slow_sc = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, kama_period))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    er = np.zeros_like(close_1d)
    er[kama_period:] = change[kama_period:] / volatility[kama_period:]
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[kama_period] = close_1d[kama_period]
    for i in range(kama_period + 1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate KAMA slope (1-period change)
    kama_slope = np.diff(kama, prepend=np.nan)
    
    # 1d RSI(14)
    rsi_period = 14
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
    
    # Align KAMA slope and RSI to 12h timeframe
    kama_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_slope)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.2x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kama_period, rsi_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_slope_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.2 * vol_ma[i]
        
        if position == 0 and in_session:
            # Long: KAMA slope positive AND RSI > 50 AND volume
            if kama_slope_aligned[i] > 0 and rsi_1d_aligned[i] > 50 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: KAMA slope negative AND RSI < 50 AND volume
            elif kama_slope_aligned[i] < 0 and rsi_1d_aligned[i] < 50 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA slope negative OR RSI < 40
            if kama_slope_aligned[i] < 0 or rsi_1d_aligned[i] < 40:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA slope positive OR RSI > 60
            if kama_slope_aligned[i] > 0 or rsi_1d_aligned[i] > 60:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_With_1d_RSI_Confirmation_v1"
timeframe = "12h"
leverage = 1.0