#!/usr/bin/env python3
"""
4H RSI Pullback with 1D Trend Filter and Volume Confirmation
Long when RSI(14) < 40, price > 1D EMA(50), and volume > 1.5x average
Short when RSI(14) > 60, price < 1D EMA(50), and volume > 1.5x average
Exit when RSI crosses 50 (mean reversion exit)
Uses RSI mean reversion in trending markets with volume confirmation to avoid false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_1d_trend_volume_v1"
timeframe = "4h"
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
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1D trend filter (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (overbought)
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (oversold)
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: above average
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: RSI pullback with 1D trend filter and volume confirmation
            if rsi[i] < 40 and close[i] > ema_1d_aligned[i]:
                # Oversold in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif rsi[i] > 60 and close[i] < ema_1d_aligned[i]:
                # Overbought in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals