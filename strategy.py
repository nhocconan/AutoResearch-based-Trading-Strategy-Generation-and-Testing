#!/usr/bin/env python3
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
    
    # Get 4h data for trend and volatility
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ATR for volatility filter (14-period)
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h EMA200 for long-term trend filter
    ema200_1h = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1h volatility ratio (current ATR / 20-period average ATR)
    tr1h = np.abs(high - low)
    tr2h = np.abs(high - np.roll(close, 1))
    tr3h = np.abs(low - np.roll(close, 1))
    tr1h[0] = np.nan
    tr2h[0] = np.nan
    tr3h[0] = np.nan
    tr_h = np.maximum(tr1h, np.maximum(tr2h, tr3h))
    atr_1h = pd.Series(tr_h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma_1h = pd.Series(atr_1h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = atr_1h / atr_ma_1h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema200_1h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility filter: avoid choppy markets
        if vol_ratio[i] < 0.8:
            signals[i] = 0.0
            position = 0
            continue
        
        # Long condition: price above 4h EMA50 and 1h EMA200, with volatility expansion
        if (close[i] > ema50_4h_aligned[i] and 
            close[i] > ema200_1h[i] and 
            vol_ratio[i] > 1.2):
            signals[i] = 0.20
            position = 1
        # Short condition: price below 4h EMA50 and 1h EMA200, with volatility expansion
        elif (close[i] < ema50_4h_aligned[i] and 
              close[i] < ema200_1h[i] and 
              vol_ratio[i] > 1.2):
            signals[i] = -0.20
            position = -1
        # Exit conditions: price crosses back below/above EMA50
        elif position == 1 and close[i] < ema50_4h_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema50_4h_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_EMA50_EMA200_VolFilter_4hTrend_v1"
timeframe = "1h"
leverage = 1.0