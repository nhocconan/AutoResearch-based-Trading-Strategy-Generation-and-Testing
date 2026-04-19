#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_AdaptiveKeltnerBreakout_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(10)
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(low_1d[1:], high_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1d EMA(20) for trend
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate dynamic Keltner channels: EMA20 ± 2*ATR10
    upper_keltner = ema_20 + 2.0 * atr_10
    lower_keltner = ema_20 - 2.0 * atr_10
    
    # Align to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume spike filter (6-period EMA on 6h data)
    volume_ema = pd.Series(volume).ewm(span=6, adjust=False, min_periods=6).mean().values
    volume_spike = volume > (1.5 * volume_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
            
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long: close breaks above upper Keltner + volume + price > EMA (uptrend)
            if close[i] > upper_aligned[i] and vol_confirm and close[i] > ema_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below lower Keltner + volume + price < EMA (downtrend)
            elif close[i] < lower_aligned[i] and vol_confirm and close[i] < ema_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Exit long: close below EMA (trend change) OR opposite Keltner touch
            if close[i] < ema_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Exit short: close above EMA (trend change) OR opposite Keltner touch
            if close[i] > ema_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals