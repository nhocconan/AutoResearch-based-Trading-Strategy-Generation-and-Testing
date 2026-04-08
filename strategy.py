#!/usr/bin/env python3
# 4h_1d_donchian_breakout_v1
# Hypothesis: 4-hour Donchian(20) breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above 20-period high AND price above 1-day EMA50 AND volume > 1.5x average volume.
# Short when price breaks below 20-period low AND price below 1-day EMA50 AND volume > 1.5x average volume.
# Exit when price crosses 1-day EMA50 or ATR-based stoploss is hit.
# Uses volume confirmation to avoid false breakouts and EMA50 for trend filter.
# Designed to generate ~20-40 trades/year to avoid fee decay while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest_high[i] = np.max(high[i - period + 1:i + 1])
        lowest_low[i] = np.min(low[i - period + 1:i + 1])
    
    # Average volume (20-period)
    avg_volume = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= period:
            vol_sum -= volume[i - period]
        if i >= period - 1:
            avg_volume[i] = vol_sum / period
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_50_1d[i] = close_1d[i]
        else:
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # ATR (14-period) for stoploss
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    atr_sum = 0.0
    for i in range(n):
        atr_sum += tr[i]
        if i >= 14:
            atr_sum -= tr[i-14]
        if i >= 13:
            atr[i] = atr_sum / 14
    
    # Align 1d EMA50 and ATR to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        highest = highest_high[i]
        lowest = lowest_low[i]
        ema_50 = ema_50_1d_aligned[i]
        atr_val = atr_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 1-day EMA50 or ATR stoploss hit
            if price < ema_50 or price < ema_50 - 2.0 * atr_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 1-day EMA50 or ATR stoploss hit
            if price > ema_50 or price > ema_50 + 2.0 * atr_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Donchian breakout with trend filter and volume confirmation
            vol_threshold = 1.5 * avg_volume[i]
            # Bullish: price breaks above 20-period high AND price above 1-day EMA50 AND volume confirmation
            if price > highest and price > ema_50 and vol > vol_threshold:
                position = 1
                signals[i] = 0.25
            # Bearish: price breaks below 20-period low AND price below 1-day EMA50 AND volume confirmation
            elif price < lowest and price < ema_50 and vol > vol_threshold:
                position = -1
                signals[i] = -0.25
    
    return signals