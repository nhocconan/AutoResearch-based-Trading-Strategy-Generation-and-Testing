#!/usr/bin/env python3
"""
4h_rsi_pullback_1d_trend_volume_v1
Hypothesis: On 4h timeframe, enter long when RSI(14) pulls back to 40-50 during 1d uptrend with volume confirmation, and short when RSI pulls back to 50-60 during 1d downtrend. Uses 1d trend as filter to avoid counter-trend trades, targeting mean reversion within the trend. Volume filter ensures momentum behind the move. Designed for 20-50 trades/year to minimize fee drag while capturing trend continuation moves.
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
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average for confirmation
    vol_ma_period = 20
    vol_ma = np.zeros(n)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    # Load 1d trend data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend
    ema_period = 50
    close_1d = df_1d['close'].values
    ema_1d = np.zeros(len(close_1d))
    ema_1d[0] = close_1d[0]
    alpha = 2 / (ema_period + 1)
    
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Determine 1d trend: price above/below EMA
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1=uptrend, -1=downtrend
    
    # Align 1d trend to 4h
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(rsi_period, vol_ma_period), n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI reaches 60 (overbought in uptrend) or trend turns down
            if rsi[i] >= 60 or trend_1d_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI reaches 40 (oversold in downtrend) or trend turns up
            if rsi[i] <= 40 or trend_1d_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume must confirm
            if not volume_ok:
                signals[i] = 0.0
                continue
                
            if trend_1d_aligned[i] == 1:  # 1d uptrend
                # Long entry: RSI pullback to 40-50
                if 40 <= rsi[i] <= 50:
                    position = 1
                    signals[i] = 0.25
            else:  # 1d downtrend
                # Short entry: RSI pullback to 50-60
                if 50 <= rsi[i] <= 60:
                    position = -1
                    signals[i] = -0.25
    
    return signals