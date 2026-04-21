#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA13 and EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # 1d EMA13 for Elder Ray Power calculations
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_14_1d / atr_ma_20_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Volume confirmation: volume / 20-period average volume (1d)
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema13 = ema_13_1d_aligned[i]
        ema21 = ema_21_1d_aligned[i]
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        atr_ratio_val = atr_ratio_1d_aligned[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 (bulls in control), price above EMA21 (uptrend), volume spike
            if (bull_power > 0 and 
                price_close > ema21 and 
                vol_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 (bears in control), price below EMA13 (downtrend), volume spike
            elif (bear_power < 0 and 
                  price_close < ema13 and 
                  vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Power reverses or price crosses opposite EMA
            if position == 1 and (bull_power <= 0 or price_close < ema13):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (bear_power >= 0 or price_close > ema21):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_Power_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0