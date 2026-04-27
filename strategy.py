#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Range_200MA_v1
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) direction on 12h timeframe
combined with RSI range filter and 200-period MA trend filter on daily timeframe.
Designed to capture trending moves while avoiding choppy markets, suitable for both
bull and bear markets by following the daily trend. Targets 12-37 trades per year
to minimize fee drag.
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
    
    # Get daily data for trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 12h data (using close prices)
    # KAMA parameters: ER period=10, Fast SC=2, Slow SC=30
    def kama(close_prices, er_period=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama_vals = np.zeros_like(close_prices)
        kama_vals[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close_prices[i] - kama_vals[i-1])
        return kama_vals
    
    # Calculate 12h KAMA
    kama_12h = kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Calculate daily RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate daily 200-period MA
    ma_200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    
    # Align daily indicators to 12h timeframe
    kama_12h_aligned = align_htf_to_ltf(prices, df_1d, kama_12h)  # No extra delay needed for KAMA
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ma_200_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ma_200_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_12h_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        ma_200_val = ma_200_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price above KAMA, RSI in neutral/bullish range (40-80), above MA200, with volume
            if close_val > kama_val and 40 <= rsi_val <= 80 and close_val > ma_200_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: price below KAMA, RSI in neutral/bearish range (20-60), below MA200, with volume
            elif close_val < kama_val and 20 <= rsi_val <= 60 and close_val < ma_200_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI becomes overbought
            if close_val < kama_val or rsi_val > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI becomes oversold
            if close_val > kama_val or rsi_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_Direction_RSI_Range_200MA_v1"
timeframe = "12h"
leverage = 1.0