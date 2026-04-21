#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Elder Ray and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA13 and EMA26 for Elder Ray components
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    ema_26_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_26_1d)
    
    # 1d ATR(22) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_22 = pd.Series(tr).rolling(window=22, min_periods=22).mean().values
    atr_ma_50 = pd.Series(atr_22).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_22 / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: volume / 30-period average volume (1d)
    vol_ma_30 = pd.Series(df_1d['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_30
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(ema_26_1d_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        bull_power = price_close - ema_13_1d_aligned[i]
        bear_power = ema_26_1d_aligned[i] - price_close
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.3
        atr_ratio_val = atr_ratio_aligned[i]
        
        if position == 0:
            # Enter long: bull power > 0, bear power < bull power, volume spike, moderate volatility
            if (bull_power > 0 and 
                bear_power < bull_power and 
                vol_ratio > vol_threshold and 
                atr_ratio_val > 0.7 and atr_ratio_val < 2.2):
                signals[i] = 0.25
                position = 1
            # Enter short: bear power > 0, bull power < bear power, volume spike, moderate volatility
            elif (bear_power > 0 and 
                  bull_power < bear_power and 
                  vol_ratio > vol_threshold and 
                  atr_ratio_val > 0.7 and atr_ratio_val < 2.2):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: power reversal or volatility extremes
            if position == 1 and (bull_power <= 0 or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (bear_power <= 0 or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_Power_1dTrend_VolumeATR"
timeframe = "6h"
leverage = 1.0