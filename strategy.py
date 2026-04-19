#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_Trend"
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
    
    # Get 1d data for KAMA and RSI (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (adaptive moving average) on 1d
    # ER = Efficiency Ratio: |close - close_prev| / sum(|change|) over period
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    direction = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    direction_sum = pd.Series(direction).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility > 0, direction_sum / volatility, 0)
    # Smoothing constants: fast = 2/(2+1), slow = 2/(30+1)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI to 4h timeframe
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: current volume > 1.5x 20-period average (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama_val = kama_4h[i]
        rsi_val = rsi_4h[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        kama_trend = price > kama_val  # Price above KAMA = uptrend
        
        if position == 0:
            # Long: Price above KAMA + RSI not overbought + volume confirmation
            if kama_trend and rsi_val < 70 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + RSI not oversold + volume confirmation
            elif not kama_trend and rsi_val > 30 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below KAMA
            if price < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above KAMA
            if price > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals