#!/usr/bin/env python3
name = "4h_1d_KAMA_RSI_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA parameters
    er_window = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, n=er_window))
    volatility_sum = pd.Series(volatility).rolling(window=er_window, min_periods=er_window).sum().values
    er = np.divide(change, volatility_sum, out=np.zeros_like(change), where=volatility_sum!=0)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Daily RSI(14)
    delta = np.diff(df_1d['close'], prepend=df_1d['close'].iloc[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Chopiness Index (14) on daily
    atr = np.zeros(len(df_1d))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 6)  # Wait for chop and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, chop > 61.8 (ranging), volume spike
            vol_condition = volume[i] > vol_ma_6[i] * 2.0
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and chop_aligned[i] > 61.8 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, chop > 61.8 (ranging), volume spike
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and chop_aligned[i] > 61.8 and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price below KAMA or RSI < 40 or chop < 38.2 (trending)
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price above KAMA or RSI > 60 or chop < 38.2 (trending)
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h KAMA with RSI and Chop filter for mean reversion in ranging markets
# - KAMA adapts to market noise, effective in both trending and ranging conditions
# - Long when price > KAMA, RSI > 50, chop > 61.8 (ranging), volume spike
# - Short when price < KAMA, RSI < 50, chop > 61.8 (ranging), volume spike
# - Chop filter ensures we only trade in ranging markets (chop > 61.8)
# - Exit when price crosses KAMA, RSI reverses, or market starts trending (chop < 38.2)
# - Volume spike (2x average) confirms institutional participation
# - Works in both bull (buy dips in uptrend ranging) and sell (sell rallies in downtrend ranging)
# - Position size 0.25 targets ~30-50 trades/year, avoiding fee drag