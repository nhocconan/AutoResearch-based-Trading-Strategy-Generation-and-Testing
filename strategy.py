#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Trend_VolumeBreakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1w: KAMA for trend filter ===
    close_1w = df_1w['close'].values
    # Calculate KAMA(30, 2, 30) - ER period=10, fast=2, slow=30
    kama_period = 30
    fast_sc = 2/(2+1)
    slow_sc = 2/(30+1)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)  # will fix in loop
    
    # Calculate ER and KAMA properly
    er = np.full_like(close_1w, np.nan)
    kama = np.full_like(close_1w, np.nan)
    
    for i in range(10, len(close_1w)):
        if i >= 10:
            change_val = np.abs(close_1w[i] - close_1w[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close_1w[i-9:i+1])))
            if volatility_sum > 0:
                er[i] = change_val / volatility_sum
            else:
                er[i] = 0
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            if np.isnan(kama[i-1]):
                kama[i] = close_1w[i]
            else:
                kama[i] = kama[i-1] + sc * (close_1w[i] - kama[i-1])
    
    # Align KAMA to 1d
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # === 1d: Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stop loss
    high = prices['high'].values
    low = prices['low'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        kama_val = kama_1w_aligned[i]
        current_vol_ma = vol_ma[i]
        current_volume = volume[i]
        current_close = close[i]
        current_atr = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(current_vol_ma) or 
            np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = current_volume > 1.5 * current_vol_ma
        
        if position == 0:
            # Long: close > 1w KAMA (uptrend) + volume breakout
            if current_close > kama_val and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: close < 1w KAMA (downtrend) + volume breakout
            elif current_close < kama_val and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: close < 1w KAMA OR stop loss
            if current_close < kama_val or current_close < entry_price - 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close > 1w KAMA OR stop loss
            if current_close > kama_val or current_close > entry_price + 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals