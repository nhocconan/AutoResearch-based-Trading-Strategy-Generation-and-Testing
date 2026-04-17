#!/usr/bin/env python3
"""
6h_1dPivot_R1S1_Volume_RSI_1dEMA50
Long: Price breaks above R1 + volume > 1.5x 6h volume MA + RSI < 30 (oversold) + price > 1D EMA50
Short: Price breaks below S1 + volume > 1.5x 6h volume MA + RSI > 70 (overbought) + price < 1D EMA50
Exit: Opposite break of R1/S1
Uses 1D RSI for mean reversion edge and 1D EMA50 for trend filter
Target: 15-25 trades/year per symbol
"""

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
    
    # Get 1D data for pivot points and filters
    df_1d = get_htf_data(prices, '1d')
    # Calculate daily pivot points
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pivot - df_1d['low']
    s1 = 2 * pivot - df_1d['high']
    # 1D RSI
    delta = df_1d['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # 1D EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h volume moving average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma[i]
        
        if position == 0:
            # Long: break above R1 + volume + RSI oversold + 1D trend
            if price > r1_aligned[i] and vol > 1.5 * vol_ma and rsi_aligned[i] < 30 and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below S1 + volume + RSI overbought + 1D trend
            elif price < s1_aligned[i] and vol > 1.5 * vol_ma and rsi_aligned[i] > 70 and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below S1
            if price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above R1
            if price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dPivot_R1S1_Volume_RSI_1dEMA50"
timeframe = "6h"
leverage = 1.0