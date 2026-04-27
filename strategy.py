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
    open_time = prices['open_time'].values
    
    # Session filter: 8-20 UTC (pre-trade to NY close)
    hours = pd.DatetimeIndex(open_time).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day RSI (14-period) using Wilder's smoothing
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    # Initial average
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        
        # Wilder's smoothing
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi_1d[i] = 100 - (100 / (1 + rs))
        else:
            rsi_1d[i] = 100
    
    # Calculate 1-day EMA (50-period) for trend filter
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        alpha = 2 / (50 + 1)
        ema_50_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Calculate 1-day ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_prev)
    tr3 = np.abs(low_1d - close_1d_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Align 1d indicators to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.20  # 20% position size
    
    # Warmup period
    start_idx = max(14, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not session_mask[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.8x average volume
        vol_filter = vol_ratio > 1.8
        
        # RSI filter: avoid extreme overbought/oversold
        rsi_filter = (rsi_1d_aligned[i] > 35) & (rsi_1d_aligned[i] < 65)
        
        # Trend filter: price relative to EMA50
        above_ema = price > ema_50_1d_aligned[i]
        below_ema = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Price above EMA50 with volume and RSI not extreme
            if above_ema and vol_filter and rsi_filter:
                signals[i] = size
                position = 1
            # Short: Price below EMA50 with volume and RSI not extreme
            elif below_ema and vol_filter and rsi_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below EMA50 or extreme RSI with volume
            if below_ema or (rsi_1d_aligned[i] > 70 and vol_ratio > 2.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above EMA50 or extreme RSI with volume
            if above_ema or (rsi_1d_aligned[i] < 30 and vol_ratio > 2.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI14_EMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0