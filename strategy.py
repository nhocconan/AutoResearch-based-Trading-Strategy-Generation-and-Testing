#!/usr/bin/env python3
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
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Minimum for KAMA
        return np.zeros(n)
    
    # KAMA parameters (Erickson 1998)
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(df_1d['close'].diff(er_len))
    volatility = np.abs(df_1d['close'].diff()).rolling(window=er_len, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(df_1d['close'])
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # Align KAMA to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Get 1d RSI
    rsi_period = 14
    delta = df_1d['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to lower timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    # Get 1d Choppiness Index (using ATR)
    chop_period = 14
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = tr.rolling(window=chop_period, min_periods=chop_period).mean()
    highest_high = df_1d['high'].rolling(window=chop_period, min_periods=chop_period).max()
    lowest_low = df_1d['low'].rolling(window=chop_period, min_periods=chop_period).min()
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(chop_period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA, RSI, Chop, volume MA
    start_idx = max(14, 20)  # RSI and chop need 14 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI < 40 (oversold), chop > 61.8 (range)
            if price > kama_val and rsi_val < 40 and chop_val > 61.8 and vol_filter:
                signals[i] = size
                position = 1
            # Short: price below KAMA (downtrend), RSI > 60 (overbought), chop > 61.8 (range)
            elif price < kama_val and rsi_val > 60 and chop_val > 61.8 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI > 70
            if price <= kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI < 30
            if price >= kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0