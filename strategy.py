#!/usr/bin/env python3
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
    
    # Get daily data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate 20-period ATR on daily for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_20 = np.full(len(tr), np.nan)
    for i in range(20, len(tr)):
        atr_20[i] = np.mean(tr[i-19:i+1])
    
    # Align indicators to 6h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # Calculate 6-period RSI on 6h for entry timing
    delta_6h = np.diff(close, prepend=close[0])
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    
    avg_gain_6h = np.zeros_like(close)
    avg_loss_6h = np.zeros_like(close)
    for i in range(6, len(close)):
        if i == 5:
            avg_gain_6h[i] = np.mean(gain_6h[1:6])
            avg_loss_6h[i] = np.mean(loss_6h[1:6])
        else:
            avg_gain_6h[i] = (avg_gain_6h[i-1] * 5 + gain_6h[i]) / 6
            avg_loss_6h[i] = (avg_loss_6h[i-1] * 5 + loss_6h[i]) / 6
    
    rs_6h = np.divide(avg_gain_6h, avg_loss_6h, out=np.full_like(avg_gain_6h, np.nan), where=avg_loss_6h!=0)
    rsi_6h = 100 - (100 / (1 + rs_6h))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(rsi_14_aligned[i]) or 
            np.isnan(atr_20_aligned[i]) or 
            np.isnan(rsi_6h[i])):
            signals[i] = 0.0
            continue
        
        # HTF conditions: RSI extreme + low volatility
        rsi_extreme = (rsi_14_aligned[i] < 30) or (rsi_14_aligned[i] > 70)
        low_vol = atr_20_aligned[i] < np.nanmedian(atr_20_aligned[max(0, i-50):i+1]) if not np.isnan(np.nanmedian(atr_20_aligned[max(0, i-50):i+1])) else False
        
        # LTF entry: RSI mean reversion
        long_entry = rsi_6h[i] < 35 and rsi_extreme and rsi_14_aligned[i] < 30 and low_vol
        short_entry = rsi_6h[i] > 65 and rsi_extreme and rsi_14_aligned[i] > 70 and low_vol
        
        # Exit: RSI returns to neutral
        exit_long = position == 1 and rsi_6h[i] > 50
        exit_short = position == -1 and rsi_6h[i] < 50
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_rsi_extreme_mean_reversion"
timeframe = "6h"
leverage = 1.0