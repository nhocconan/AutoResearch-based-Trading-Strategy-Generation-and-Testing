#!/usr/bin/env python3
"""
1h RSI Pullback with 4h Trend and Daily Volume Confirmation
Long when: 4h trend up (EMA50 > EMA200), daily volume above average, and RSI pulls back to 40-50 in uptrend
Short when: 4h trend down (EMA50 < EMA200), daily volume above average, and RSI bounces to 50-60 in downtrend
Exit when RSI reaches opposite extreme (60 for long, 40 for short) or trend reverses
Uses RSI mean reversion within strong trends, filtered by higher timeframe trend and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === RSI (14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h trend filter (EMA50 and EMA200) ===
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(df_4h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # === Daily volume confirmation ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / (vol_ma_1d + 1e-10)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema200_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend
        uptrend_4h = ema50_4h_aligned[i] > ema200_4h_aligned[i]
        downtrend_4h = ema50_4h_aligned[i] < ema200_4h_aligned[i]
        
        # Volume filter: need above average daily volume
        vol_filter = vol_ratio_1d_aligned[i] > 1.1
        
        if position == 1:  # Long position
            # Exit: RSI reaches 60 or 4h trend turns down
            if rsi[i] >= 60 or not uptrend_4h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI reaches 40 or 4h trend turns up
            if rsi[i] <= 40 or not downtrend_4h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need volume filter
            if not vol_filter:
                signals[i] = 0.0
                continue
            
            # Entry: RSI pullback in trending market
            if uptrend_4h and 40 <= rsi[i] <= 50:
                # Pullback to support in uptrend -> long
                position = 1
                signals[i] = 0.20
            elif downtrend_4h and 50 <= rsi[i] <= 60:
                # Bounce to resistance in downtrend -> short
                position = -1
                signals[i] = -0.20
    
    return signals