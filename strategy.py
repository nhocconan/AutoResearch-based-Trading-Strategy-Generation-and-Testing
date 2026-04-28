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
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI(14) with proper Wilder's smoothing
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First 14-period average
    avg_loss[13] = np.mean(loss[1:14])
    # Wilder's smoothing: avg = (prev_avg * 13 + current) / 14
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    # For indices < 13, keep as 0 (will be handled by RS calculation)
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # When no losses, RSI=100
    
    # Calculate daily SMA(50)
    sma_50 = np.full_like(close_1d, np.nan)
    for i in range(49, len(close_1d)):
        sma_50[i] = np.mean(close_1d[i-49:i+1])
    
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Align daily indicators to 1h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period (need enough data for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(sma_50_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above SMA50 for long, below for short
        uptrend = close[i] > sma_50_aligned[i]
        downtrend = close[i] < sma_50_aligned[i]
        
        # RSI conditions: oversold/overbought
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Volatility filter: avoid extremely low volatility
        # Calculate 20-period ATR moving average for volatility regime
        atr_ma = np.full_like(atr, np.nan)
        for j in range(19, len(atr)):
            atr_ma[j] = np.mean(atr[j-19:j+1])
        atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
        if np.isnan(atr_ma_aligned[i]):
            signals[i] = 0.0
            continue
        vol_ok = atr_aligned[i] > (atr_ma_aligned[i] * 0.3)
        
        # Long conditions: uptrend + RSI oversold + volatility ok
        long_condition = uptrend and rsi_oversold and vol_ok
        
        # Short conditions: downtrend + RSI overbought + volatility ok
        short_condition = downtrend and rsi_overbought and vol_ok
        
        # Entry logic with position sizing
        if long_condition and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.20
            position = -1
        # Exit conditions: opposite RSI extreme (more conservative)
        elif position == 1 and rsi_aligned[i] > 50:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi_aligned[i] < 50:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_RSI_SMA50_Trend_Filter"
timeframe = "1h"
leverage = 1.0