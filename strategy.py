#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_Filter_RSI_Entry"
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
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA on daily: ER = |close - close[10]| / sum(|close - close[-1]| over 10)
    change = np.abs(close_1d - np.roll(close_1d, 10))
    change[np.isnan(change)] = 0
    volatility = np.abs(np.diff(close_1d, prepend=np.nan))
    volatility = np.concatenate([[np.nan], volatility[:-1]])
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(vol_sum != 0, change / vol_sum, 0)
    # Smooth constants: fast = 2/(2+1), slow = 2/(30+1)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(close_1d[i]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 4h
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI on 4h
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Time filter: 08-20 UTC (active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if not time_filter[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(kama_4h[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price above KAMA + RSI < 40 (oversold in uptrend)
            if price > kama_4h[i] and rsi[i] < 40 and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + RSI > 60 (overbought in downtrend)
            elif price < kama_4h[i] and rsi[i] > 60 and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA OR RSI > 70 (overbought)
            if price < kama_4h[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA OR RSI < 30 (oversold)
            if price > kama_4h[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals