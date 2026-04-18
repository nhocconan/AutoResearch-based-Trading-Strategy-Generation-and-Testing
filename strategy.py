#!/usr/bin/env python3
"""
12h_Momentum_1wTrend_1dVolume
Hypothesis: Trade momentum on 12h using 1w trend filter (bull/bear) with 1d volume confirmation. 
In bull (price > 1w EMA50): long when 12h RSI crosses above 50 + volume > 1.5x 24-period average.
In bear (price < 1w EMA50): short when 12h RSI crosses below 50 + volume > 1.5x 24-period average.
Uses 1w EMA50 for trend filter to avoid counter-trend trades, and volume confirmation to ensure 
momentum validity. Designed for low trade frequency (<30/year) to minimize fee drag. Works in 
both bull and bear by following the higher-timeframe trend.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA50
    ema_period = 50
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align 1w EMA50 to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 1d volume 24-period average
    vol_ma_period = 24
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= vol_ma_period:
        for i in range(vol_ma_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i - vol_ma_period:i])
    
    # Align 1d volume MA to 12h
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 12h RSI(14)
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(close) >= rsi_period + 1:
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        for i in range(rsi_period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.full_like(close, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_period, vol_ma_period, rsi_period + 1)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend: bull if price > 1w EMA50, bear if price < 1w EMA50
        is_bull = close[i] > ema_1w_aligned[i]
        is_bear = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long in bull: RSI crosses above 50 + volume
            if i > 0 and not np.isnan(rsi[i-1]) and rsi[i-1] <= 50 and rsi[i] > 50 and is_bull and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short in bear: RSI crosses below 50 + volume
            elif i > 0 and not np.isnan(rsi[i-1]) and rsi[i-1] >= 50 and rsi[i] < 50 and is_bear and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses below 50 or trend turns bear
            if (i > 0 and not np.isnan(rsi[i-1]) and rsi[i-1] > 50 and rsi[i] <= 50) or not is_bull:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses above 50 or trend turns bull
            if (i > 0 and not np.isnan(rsi[i-1]) and rsi[i-1] < 50 and rsi[i] >= 50) or not is_bear:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Momentum_1wTrend_1dVolume"
timeframe = "12h"
leverage = 1.0