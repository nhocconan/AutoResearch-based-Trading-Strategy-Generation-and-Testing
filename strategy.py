#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_Follow"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on 1d close
    er_len = 10
    fast_ema = 2 / (2 + 1)
    slow_ema = 2 / (30 + 1)
    
    change = np.abs(np.diff(close_1d, n=er_len))
    change = np.concatenate([np.full(er_len, np.nan), change])
    
    volatility = np.abs(np.diff(close_1d))
    volatility = np.concatenate([np.array([np.nan]), volatility])
    vol_sum = pd.Series(volatility).rolling(window=er_len, min_periods=er_len).sum().values
    
    er = np.divide(change, vol_sum, out=np.full_like(change, np.nan), where=vol_sum!=0)
    sc = np.square(er * (fast_ema - slow_ema) + slow_ema)
    
    kama = np.full_like(close_1d, np.nan)
    kama[er_len] = close_1d[er_len]
    for i in range(er_len + 1, len(close_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 4h EMA20 for entry filter
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 14)
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(ema20[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama_val = kama_aligned[i]
        ema_val = ema20[i]
        atr_val = atr[i]
        
        # Volume filter
        volume_ok = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: price above KAMA and above EMA20 with volume
            if price > kama_val and price > ema_val and volume_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below KAMA and below EMA20 with volume
            elif price < kama_val and price < ema_val and volume_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: price crosses below KAMA or stop loss hit
            if price < kama_val or price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA or stop loss hit
            if price > kama_val or price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals