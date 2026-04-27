#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[:13]) if len(gain) >= 13 else np.nan
    avg_loss[13] = np.mean(loss[:13]) if len(loss) >= 13 else np.nan
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([[np.nan], rsi_1d])  # align with close_1d index
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = np.zeros(len(tr_1d))
    for i in range(len(tr_1d)):
        if i < 13:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Align daily RSI and ATR to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get weekly data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA(20)
    ema_1w_20 = np.zeros(len(close_1w))
    alpha = 2 / (20 + 1)
    ema_1w_20[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_1w_20[i] = close_1w[i] * alpha + ema_1w_20[i-1] * (1 - alpha)
    
    # Align weekly EMA20 to 12h timeframe
    ema_1w_20_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_20)
    
    # Calculate 12h RSI(14) for momentum filter
    delta_12h = np.diff(close)
    gain_12h = np.where(delta_12h > 0, delta_12h, 0.0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0.0)
    avg_gain_12h = np.zeros(n)
    avg_loss_12h = np.zeros(n)
    avg_gain_12h[13] = np.mean(gain_12h[:13]) if n >= 13 else np.nan
    avg_loss_12h[13] = np.mean(loss_12h[:13]) if n >= 13 else np.nan
    for i in range(14, n):
        avg_gain_12h[i] = (avg_gain_12h[i-1] * 13 + gain_12h[i-1]) / 14
        avg_loss_12h[i] = (avg_loss_12h[i-1] * 13 + loss_12h[i-1]) / 14
    rs_12h = np.where(avg_loss_12h != 0, avg_gain_12h / avg_loss_12h, 0)
    rsi_12h = 100 - (100 / (1 + rs_12h))
    
    # Calculate volume average (10-period)
    vol_ma_10 = np.full(n, np.nan)
    for i in range(10, n):
        vol_ma_10[i] = np.mean(volume[i-10:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(14, 10)  # RSI needs 14, volume MA needs 10
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_1w_20_aligned[i]) or
            np.isnan(rsi_12h[i]) or
            np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_10[i] if vol_ma_10[i] > 0 else 0
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_filter = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        if position == 0:
            # Long: price above weekly EMA20, RSI neutral, volume confirmation
            if (price > ema_1w_20_aligned[i] and 
                rsi_filter and 
                volume_confirmation and 
                rsi_12h[i] > 50):  # additional momentum filter
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA20, RSI neutral, volume confirmation
            elif (price < ema_1w_20_aligned[i] and 
                  rsi_filter and 
                  volume_confirmation and 
                  rsi_12h[i] < 50):  # additional momentum filter
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below weekly EMA20 or RSI overbought
            if (price < ema_1w_20_aligned[i] or 
                rsi_1d_aligned[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above weekly EMA20 or RSI oversold
            if (price > ema_1w_20_aligned[i] or 
                rsi_1d_aligned[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_weekly_EMA20_dailyRSI_volume_filter_v1"
timeframe = "12h"
leverage = 1.0