#!/usr/bin/env python3
"""
1h_4h_1d_RSI_MeanReversion_TrendFilter
Hypothesis: Use 4h RSI for trend direction (bull/bear regime) and 1d RSI for mean reversion opportunities.
In bull markets (4h RSI > 50), buy near 1d RSI oversold (<30) with volume confirmation.
In bear markets (4h RSI < 50), sell near 1d RSI overbought (>70) with volume confirmation.
Trades only during 08-20 UTC to avoid low-volume periods. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h and 1d data once
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 14 or len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI for trend filter
    close_4h = df_4h['close'].values
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    
    avg_gain_4h = np.zeros_like(gain_4h)
    avg_loss_4h = np.zeros_like(loss_4h)
    for i in range(len(gain_4h)):
        if i < 14:
            if i > 0:
                avg_gain_4h[i] = np.mean(gain_4h[:i+1])
                avg_loss_4h[i] = np.mean(loss_4h[:i+1])
            else:
                avg_gain_4h[i] = gain_4h[i]
                avg_loss_4h[i] = loss_4h[i]
        else:
            avg_gain_4h[i] = (avg_gain_4h[i-1] * 13 + gain_4h[i]) / 14
            avg_loss_4h[i] = (avg_loss_4h[i-1] * 13 + loss_4h[i]) / 14
    
    rs_4h = np.divide(avg_gain_4h, avg_loss_4h, out=np.full_like(avg_gain_4h, 50.0), where=avg_loss_4h!=0)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1d RSI for mean reversion signals
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    
    avg_gain_1d = np.zeros_like(gain_1d)
    avg_loss_1d = np.zeros_like(loss_1d)
    for i in range(len(gain_1d)):
        if i < 14:
            if i > 0:
                avg_gain_1d[i] = np.mean(gain_1d[:i+1])
                avg_loss_1d[i] = np.mean(loss_1d[:i+1])
            else:
                avg_gain_1d[i] = gain_1d[i]
                avg_loss_1d[i] = loss_1d[i]
        else:
            avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
            avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
    
    rs_1d = np.divide(avg_gain_1d, avg_loss_1d, out=np.full_like(avg_gain_1d, 50.0), where=avg_loss_1d!=0)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 24-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 24:
            volume_avg[i] = np.mean(volume[i-24:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if NaN in critical values
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not (session_filter[i] and volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_4h_val = rsi_4h_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        price = close[i]
        
        if position == 0:
            # Bull market (4h RSI > 50): look for long at 1d RSI oversold
            if rsi_4h_val > 50 and rsi_1d_val < 30:
                signals[i] = 0.20
                position = 1
            # Bear market (4h RSI < 50): look for short at 1d RSI overbought
            elif rsi_4h_val < 50 and rsi_1d_val > 70:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 1d RSI returns to neutral or 4h trend turns bearish
            if rsi_1d_val >= 50 or rsi_4h_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 1d RSI returns to neutral or 4h trend turns bullish
            if rsi_1d_val <= 50 or rsi_4h_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_RSI_MeanReversion_TrendFilter"
timeframe = "1h"
leverage = 1.0