#!/usr/bin/env python3
"""
1h_4h1d_RSI_Trend_Filter
Hypothesis: Uses 4h RSI for trend bias (above/below 50) and 1d RSI for momentum confirmation.
Trades long when 4h RSI > 50 and 1d RSI > 55 (bullish momentum) and price pulls back to 1h EMA(21).
Trades short when 4h RSI < 50 and 1d RSI < 45 (bearish momentum) and price pulls back to 1h EMA(21).
Designed for both bull and bear markets by aligning with higher timeframe momentum.
Target: 20-40 trades/year on 1h.
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
    
    # Get HTF data once
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h RSI for trend bias
    close_4h = df_4h['close'].values
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = np.full_like(close_4h, np.nan)
    avg_loss_4h = np.full_like(close_4h, np.nan)
    for i in range(14, len(close_4h)):
        if i == 14:
            avg_gain_4h[i] = np.mean(gain_4h[1:15])
            avg_loss_4h[i] = np.mean(loss_4h[1:15])
        else:
            avg_gain_4h[i] = (avg_gain_4h[i-1] * 13 + gain_4h[i]) / 14
            avg_loss_4h[i] = (avg_loss_4h[i-1] * 13 + loss_4h[i]) / 14
    rs_4h = np.where(avg_loss_4h != 0, avg_gain_4h / avg_loss_4h, 0)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1d RSI for momentum confirmation
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = np.full_like(close_1d, np.nan)
    avg_loss_1d = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain_1d[i] = np.mean(gain_1d[1:15])
            avg_loss_1d[i] = np.mean(loss_1d[1:15])
        else:
            avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
            avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
    rs_1d = np.where(avg_loss_1d != 0, avg_gain_1d / avg_loss_1d, 0)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1h EMA(21) for pullback entries
    close_series = pd.Series(close)
    ema_21 = close_series.ewm(span=21, adjust=False).values
    
    # Volume filter: current volume > 1.5 x 20-period average (moderate)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure EMA and HTF data available
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_21[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: 4h RSI > 50 (bullish trend), 1d RSI > 55 (bullish momentum), 
            # price at or near EMA(21) pullback, with volume
            if (rsi_4h_aligned[i] > 50 and 
                rsi_1d_aligned[i] > 55 and 
                close[i] <= ema_21[i] * 1.01 and  # Allow 1% above EMA for entry
                vol_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: 4h RSI < 50 (bearish trend), 1d RSI < 45 (bearish momentum), 
            # price at or near EMA(21) pullback, with volume
            elif (rsi_4h_aligned[i] < 50 and 
                  rsi_1d_aligned[i] < 45 and 
                  close[i] >= ema_21[i] * 0.99 and  # Allow 1% below EMA for entry
                  vol_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: 4h RSI < 45 (loss of bullish momentum) or price extends too far from EMA
            if (rsi_4h_aligned[i] < 45 or 
                close[i] > ema_21[i] * 1.03):  # Exit if 3% above EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h RSI > 55 (loss of bearish momentum) or price extends too far from EMA
            if (rsi_4h_aligned[i] > 55 or 
                close[i] < ema_21[i] * 0.97):  # Exit if 3% below EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_RSI_Trend_Filter"
timeframe = "1h"
leverage = 1.0