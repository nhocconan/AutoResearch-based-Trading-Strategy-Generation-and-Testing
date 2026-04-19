#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1dKeltnerBreakout_VolumeTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner channels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA20 for Keltner basis
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    # 1d ATR(10) for Keltner width
    tr1 = np.maximum(df_1d['high'][1:] - df_1d['low'][1:], np.abs(df_1d['high'][1:] - df_1d['close'][:-1]))
    tr1 = np.maximum(tr1, np.abs(df_1d['low'][1:] - df_1d['close'][:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr10_1d = pd.Series(tr1).ewm(span=10, adjust=False, min_periods=10).mean().values
    # Keltner channels
    upper_1d = ema20_1d + 2.0 * atr10_1d
    lower_1d = ema20_1d - 2.0 * atr10_1d
    # Align to 12h
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h volume confirmation: current > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_1d_aligned[i]
        lower = lower_1d_aligned[i]
        ema50 = ema50_1d_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above upper Keltner with volume and uptrend
            if price > upper and volume_confirmed and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Keltner with volume and downtrend
            elif price < lower and volume_confirmed and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below EMA20 or ATR-based stop
            if price < ema20_1d_aligned[i] or price < close[i-1] - 1.5 * atr10_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above EMA20 or ATR-based stop
            if price > ema20_1d_aligned[i] or price > close[i-1] + 1.5 * atr10_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals