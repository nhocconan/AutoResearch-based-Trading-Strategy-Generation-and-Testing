#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# Uses 4h EMA50 for trend direction, 1h RSI(14) for mean reversion signals,
# and volume spikes (1.5x 20-period average) to confirm entries.
# In uptrend (price > 4h EMA50): long when RSI < 30, exit when RSI > 50
# In downtrend (price < 4h EMA50): short when RSI > 70, exit when RSI < 50
# Works in both bull and bear markets by following 4h trend while buying dips/selling rallies.
# Target: 20-40 trades/year to minimize fee decay while capturing mean reversion within trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 4h
    close_4h = df_4h['close'].values
    ema_len = 50
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_4h[ema_len-1] = np.mean(close_4h[:ema_len])
        for i in range(ema_len, len(close_4h)):
            ema_4h[i] = (close_4h[i] * multiplier) + (ema_4h[i-1] * (1 - multiplier))
    
    # Calculate 1h RSI(14)
    rsi_len = 14
    rsi = np.full(n, np.nan)
    if n >= rsi_len:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[rsi_len] = np.mean(gain[:rsi_len])
        avg_loss[rsi_len] = np.mean(loss[:rsi_len])
        
        for i in range(rsi_len + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_len - 1) + gain[i-1]) / rsi_len
            avg_loss[i] = (avg_loss[i-1] * (rsi_len - 1) + loss[i-1]) / rsi_len
        
        rs = np.where(avg_loss[rsi_len:] != 0, avg_gain[rsi_len:] / avg_loss[rsi_len:], 0)
        rsi[rsi_len:] = 100 - (100 / (1 + rs))
    
    # Calculate 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align 4h EMA to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.20
    
    # Warmup period
    start_idx = max(50, rsi_len, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_4h_aligned[i]
        rsi_val = rsi[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: at least 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: Uptrend + RSI oversold + volume
            if price > ema_trend and rsi_val < 30 and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Downtrend + RSI overbought + volume
            elif price < ema_trend and rsi_val > 70 and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI returns to neutral or trend reversal
            if rsi_val > 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI returns to neutral or trend reversal
            if rsi_val < 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0