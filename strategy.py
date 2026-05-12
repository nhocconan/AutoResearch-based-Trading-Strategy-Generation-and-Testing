#!/usr/bin/env python3
name = "4h_Vortex_Trend_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # === DAILY TREND FILTER (1d EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VORTEX INDICATOR (14-period) ===
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align tr to original length
    
    vm_plus = np.abs(high - low[:-1])  # |high - previous low|
    vm_minus = np.abs(low - high[:-1]) # |low - previous high|
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    sum_vm_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: VI+ > VI- (bullish trend), price above daily EMA50, volume spike
            if (vi_plus[i] > vi_minus[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (bearish trend), price below daily EMA50, volume spike
            elif (vi_minus[i] > vi_plus[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: VI- > VI+ (trend turns bearish) OR price crosses below daily EMA50
            if (vi_minus[i] > vi_plus[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ > VI- (trend turns bullish) OR price crosses above daily EMA50
            if (vi_plus[i] > vi_minus[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals