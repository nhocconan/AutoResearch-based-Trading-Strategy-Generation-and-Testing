#!/usr/bin/env python3
name = "4h_KAMA_1dTrend_Volume_CR_v1"
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
    
    # Get 1d data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d ATR for volatility regime
    tr1 = np.maximum(df_1d['high'], df_1d['close'].shift(1)) - np.minimum(df_1d['low'], df_1d['close'].shift(1))
    tr1[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_14 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # KAMA parameters
    er_len = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate KAMA
    change = np.abs(np.diff(close, k=er_len))
    change = np.concatenate([np.full(er_len, np.nan), change])
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    volatility = pd.Series(volatility).rolling(window=er_len, min_periods=er_len).sum().values
    volatility = np.concatenate([np.full(er_len-1, np.nan), volatility[er_len-1:]])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]
    for i in range(er_len+1, len(close)):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 4h volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, er_len)
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(kama[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extreme volatility (ATR > 2 * 50-period average)
        atr_avg = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr_14_aligned[i] < (2 * atr_avg[i]) if not np.isnan(atr_avg[i]) else True
        
        if position == 0:
            # Long: price above KAMA + uptrend + volume + volatility filter
            if close[i] > kama[i] and close[i] > ema_34_aligned[i] and volume_filter[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + downtrend + volume + volatility filter
            elif close[i] < kama[i] and close[i] < ema_34_aligned[i] and volume_filter[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through KAMA
            if position == 1:
                if close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals