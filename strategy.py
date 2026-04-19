#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend + 1d RSI + volume filter
# KAMA adapts to market noise, reducing whipsaw in sideways markets
# 1d RSI filters for overbought/oversold conditions on higher timeframe
# Volume confirmation ensures institutional participation
# Designed to work in both trending and ranging markets with fewer trades
# Target: 20-30 trades/year to avoid fee drag
name = "4h_KAMA_RSI_1dVolume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    
    # 1d RSI (14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d Volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4h KAMA (ER=10, SC=2,30)
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1) if len(close.shape) > 1 else np.array([np.sum(np.abs(np.diff(close[i-er_length:i+1]))) if i >= er_length else 0 for i in range(len(close))])
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_val = kama(close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_val[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 1d average
        if i >= 20:
            vol_ma = vol_ma_1d_aligned[i]
        else:
            vol_ma = vol_ma_1d_aligned[i] if not np.isnan(vol_ma_1d_aligned[i]) else volume[i]
        volume_filter = vol_ma > 0 and volume[i] > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price > KAMA + RSI < 40 (oversold) + volume
            if close[i] > kama_val[i] and rsi_1d_aligned[i] < 40 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price < KAMA + RSI > 60 (overbought) + volume
            elif close[i] < kama_val[i] and rsi_1d_aligned[i] > 60 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA OR RSI > 60 (overbought)
            if close[i] < kama_val[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA OR RSI < 40 (oversold)
            if close[i] > kama_val[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals