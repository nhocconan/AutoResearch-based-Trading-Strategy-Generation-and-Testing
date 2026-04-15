#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 1d timeframe (primary)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly HTF data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(close) > 1 else 0
    # Simplified ER calculation for array
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            dir_change = np.abs(close[i] - close[i-10])
            vol_sum = np.sum(np.abs(np.diff(close[i-9:i+1]))) if i >= 9 else 1
            er[i] = dir_change / (vol_sum + 1e-10)
    er = np.where(er > 1, 1, er)  # Cap at 1
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align 1d KAMA to 1d (no alignment needed as we're on 1d timeframe)
    kama_1d = kama
    
    # Calculate weekly RSI from weekly data
    weekly_close = df_1w['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    weekly_rsi = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 1d
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_1w, weekly_rsi)
    
    # Calculate 1d ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d[i]) or np.isnan(weekly_rsi_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above KAMA (bullish trend)
        # 2. Weekly RSI not overbought (< 60) 
        # 3. Volume confirmation: volume > 1.2x average
        # 4. Volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        if (close[i] > kama_1d[i] and
            weekly_rsi_aligned[i] < 60 and
            volume_ratio[i] > 1.2 and
            atr[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below KAMA (bearish trend)
        # 2. Weekly RSI not oversold (> 40)
        # 3. Volume confirmation: volume > 1.2x average
        # 4. Volatility filter: ATR > 0.3% of price
        elif (close[i] < kama_1d[i] and
              weekly_rsi_aligned[i] > 40 and
              volume_ratio[i] > 1.2 and
              atr[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_WeeklyRSI_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0