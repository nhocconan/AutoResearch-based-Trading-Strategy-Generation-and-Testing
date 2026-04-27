# 4h EMA34 Volume Filter Optimized for BTC/ETH
# Hypothesis: A 4h EMA34 trend filter with volume confirmation (1.5x average volume) 
# and minimum 6-hour hold time reduces overtrading while maintaining trend capture 
# in both bull and bear markets. The hold time prevents whipsaws and excessive 
# trading during ranging periods, addressing the overtrading failures seen in 
# recent experiments.

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on daily close for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align daily EMA to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 14-period ATR for volatility filter
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    bars_since_entry = 0
    
    # Warmup period
    start_idx = max(34, 14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            bars_since_entry = 0
            # Long: Price above daily EMA34 with volume confirmation
            if price > ema_1d_aligned[i] and vol_ratio > 1.5:
                signals[i] = size
                position = 1
            # Short: Price below daily EMA34 with volume confirmation
            elif price < ema_1d_aligned[i] and vol_ratio > 1.5:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            bars_since_entry += 1
            # Long exit: Price closes below daily EMA34 OR minimum 6 bars held
            if price < ema_1d_aligned[i] or bars_since_entry >= 6:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = size
        elif position == -1:
            bars_since_entry += 1
            # Short exit: Price closes above daily EMA34 OR minimum 6 bars held
            if price > ema_1d_aligned[i] or bars_since_entry >= 6:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_EMA34_Volume_Filter_Optimized"
timeframe = "4h"
leverage = 1.0