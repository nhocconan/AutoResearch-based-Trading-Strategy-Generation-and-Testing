#!/usr/bin/env python3
"""
1D Price Action + Volume + 1W Trend Filter
Long: Price > 1D VWAP + volume > 1.5x 1D volume MA(20) + price > 1W EMA50
Short: Price < 1D VWAP + volume > 1.5x 1D volume MA(20) + price < 1W EMA50
Exit: Price crosses back through 1D VWAP
Uses 1W EMA50 to align with longer-term trend and reduce false signals in chop
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
    
    # Get 1D data for VWAP and volume MA
    df_1d = get_htf_data(prices, '1d')
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    volume_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean()
    
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap.values)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20.values)
    
    # Get 1W EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_aligned[i]) or np.isnan(volume_ma_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vwap_val = vwap_aligned[i]
        vol_ma = volume_ma_20_aligned[i]
        ema_50 = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price above VWAP + volume confirmation + 1W uptrend
            if price > vwap_val and vol > 1.5 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP + volume confirmation + 1W downtrend
            elif price < vwap_val and vol > 1.5 * vol_ma and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below VWAP
            if price < vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above VWAP
            if price > vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1D_VWAP_Volume_1WTrend"
timeframe = "1d"
leverage = 1.0