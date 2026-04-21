#!/usr/bin/env python3
"""
4h_RSI_Recovery_Dip_Buy
Hypothesis: Buy the dip when RSI shows oversold conditions on 4h with volume confirmation and 1d EMA trend filter. 
In bear markets, RSI often drops to oversold levels during panic selling, followed by mean-reversion bounces. 
Volume spike confirms buying interest. In bull markets, same mechanism works on pullbacks to rising trend.
Uses RSI(14) < 30 for entry, exits when RSI > 70 or price closes below 1d EMA50.
Target: ~25-40 trades/year on 4h, low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === RSI(14) on 4h ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_50_1d_aligned[i]
        rsi_val = rsi[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: RSI oversold + volume spike + price above 1d EMA50 (uptrend filter)
            if (rsi_val < 30 and 
                vol_spike > 1.8 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit: RSI overbought OR price closes below 1d EMA50
            if (rsi_val > 70 or 
                price_close < trend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # hold
    
    return signals

name = "4h_RSI_Recovery_Dip_Buy"
timeframe = "4h"
leverage = 1.0