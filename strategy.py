#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily Close ===
    close_1d = df_1d['close'].values
    
    # === Daily ATR (14) for volatility regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50  # Current ATR vs 50-day average
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === Daily EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Price ===
    price_close = prices['close'].values
    price_open = prices['open'].values
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio_val = atr_ratio_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long in low volatility (contraction) + uptrend + volume
            if (atr_ratio_val < 0.8 and   # Volatility contraction (< 80% of average)
                price_close[i] > ema_trend and
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short in low volatility (contraction) + downtrend + volume
            elif (atr_ratio_val < 0.8 and   # Volatility contraction
                  price_close[i] < ema_trend and
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when volatility expands or trend reverses
            if position == 1 and (atr_ratio_val > 1.2 or price_close[i] < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (atr_ratio_val > 1.2 or price_close[i] > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Volatility_Contraction_EMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0