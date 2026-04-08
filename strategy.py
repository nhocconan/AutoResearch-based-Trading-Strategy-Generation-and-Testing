#!/usr/bin/env python3
"""
6h_1d_rsi_sma_volume_v1
Hypothesis: On 6h timeframe, buy when RSI(14) < 30 and price > SMA(50) with volume confirmation,
sell when RSI(14) > 70 and price < SMA(50) with volume confirmation. Use 1d trend filter: only take
longs when price > 1d EMA(200) and shorts when price < 1d EMA(200). This combines mean reversion
(RSI extremes) with trend filter (1d EMA) and volume confirmation to avoid false signals.
Designed to work in both bull and bear markets by using 1d trend filter to align with higher timeframe trend.
Target: 12-30 trades/year per symbol (48-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rsi_sma_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # SMA calculation
    def calculate_sma(prices, period):
        sma = np.full_like(prices, np.nan)
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        return sma
    
    # RSI and SMA
    rsi = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_200 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        # Calculate EMA
        alpha = 2 / (200 + 1)
        ema_200[199] = np.mean(close_1d[:200])  # SMA for first value
        for i in range(200, len(close_1d)):
            ema_200[i] = alpha * close_1d[i] + (1 - alpha) * ema_200[i-1]
    
    # Align 1d EMA to 6h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(sma_50[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50 or price < SMA(50)
            if rsi[i] > 50 or close[i] < sma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 or price > SMA(50)
            if rsi[i] < 50 or close[i] > sma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: RSI < 30, price > SMA(50), price > 1d EMA(200), volume confirmation
            if (rsi[i] < 30 and close[i] > sma_50[i] and 
                close[i] > ema_200_aligned[i] and vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70, price < SMA(50), price < 1d EMA(200), volume confirmation
            elif (rsi[i] > 70 and close[i] < sma_50[i] and 
                  close[i] < ema_200_aligned[i] and vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals