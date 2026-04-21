#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for trend and structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 1d ATR-based volatility threshold
    atr_14_current = pd.Series(atr_14).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_current)
    
    # Volume confirmation: volume / 30-period average volume (1d)
    vol_ma_30 = pd.Series(df_1d['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_30
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_1d_aligned[i]
        atr_current = atr_14_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        if position == 0:
            # Enter long: price > EMA34 (uptrend), volatility contraction then expansion, volume spike
            if (price_close > ema_trend and 
                atr_ratio_val < 0.8 and  # Volatility contraction
                vol_ratio > 1.5):        # Volume spike
                signals[i] = 0.25
                position = 1
            # Enter short: price < EMA34 (downtrend), volatility contraction then expansion, volume spike
            elif (price_close < ema_trend and 
                  atr_ratio_val < 0.8 and 
                  vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: volatility expansion beyond threshold or trend reversal
            if position == 1 and (atr_ratio_val > 1.5 or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (atr_ratio_val > 1.5 or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VolatilityContraction_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0