#!/usr/bin/env python3
"""
1h ADX Trend + Volume Spike with 4h RSI Filter
Trades only in strong trends (ADX > 25) with volume confirmation (2x avg volume).
Uses 4h RSI to avoid overbought/oversold extremes, improving performance in both bull and bear markets.
Designed for low trade frequency (target: 20-50 trades/year) by requiring multiple confluence factors.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ADX (14)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
    
    tr = np.maximum(np.abs(high - low), 
                    np.maximum(np.abs(high - np.roll(close, 1)), 
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get 4h data for RSI filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h RSI (14)
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Volume spike (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(rsi_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 25, +DI > -DI, RSI < 70, volume spike
            if (adx[i] > 25 and 
                plus_di[i] > minus_di[i] and 
                rsi_4h_aligned[i] < 70 and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: ADX > 25, -DI > +DI, RSI > 30, volume spike
            elif (adx[i] > 25 and 
                  minus_di[i] > plus_di[i] and 
                  rsi_4h_aligned[i] > 30 and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long: exit when ADX < 20 or -DI > +DI
            signals[i] = 0.20
            if adx[i] < 20 or minus_di[i] > plus_di[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short: exit when ADX < 20 or +DI > -DI
            signals[i] = -0.20
            if adx[i] < 20 or plus_di[i] > minus_di[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_ADX_Trend_Volume_Spike_4hRSI"
timeframe = "1h"
leverage = 1.0