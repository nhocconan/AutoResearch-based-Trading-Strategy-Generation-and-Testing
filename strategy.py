#!/usr/bin/env python3
# 4h_1d_atr_breakout_v3
# Hypothesis: 4-hour ATR breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above ATR-based upper band, volume > 1.5x average, and price above 1-day EMA50.
# Short when price breaks below ATR-based lower band, volume > 1.5x average, and price below 1-day EMA50.
# Exit when price returns to 4-period EMA.
# Uses 1-day EMA50 for trend bias to avoid counter-trend trades in bear markets.
# Designed to generate ~20-40 trades/year to avoid fee decay while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_atr_breakout_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align 1-day EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR (14-period) for volatility bands
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr[13] = np.mean(tr[1:14])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR-based bands (2.0 * ATR)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    for i in range(14, n):
        if not np.isnan(atr[i]):
            upper_band[i] = close[i-1] + 2.0 * atr[i]
            lower_band[i] = close[i-1] - 2.0 * atr[i]
    
    # Calculate 4-period EMA for exit
    ema_4 = np.full(n, np.nan)
    if n >= 4:
        ema_4[3] = np.mean(close[:4])
        alpha = 2.0 / (4 + 1)
        for i in range(4, n):
            ema_4[i] = alpha * close[i] + (1 - alpha) * ema_4[i-1]
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_4[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to 4-period EMA
            if price <= ema_4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 4-period EMA
            if price >= ema_4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper band with volume expansion and above 1d EMA50
            if price > upper_band[i] and vol_ratio > 1.5 and price > ema50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band with volume expansion and below 1d EMA50
            elif price < lower_band[i] and vol_ratio > 1.5 and price < ema50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals