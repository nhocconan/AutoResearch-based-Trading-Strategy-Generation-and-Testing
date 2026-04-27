#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter: EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_34 = np.full(len(df_1d), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_1d)):
        if i < 33:
            ema_1d_34[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_1d_34[i-1]):
                ema_1d_34[i] = np.mean(close_1d[i-33:i+1])
            else:
                ema_1d_34[i] = close_1d[i] * alpha + ema_1d_34[i-1] * (1 - alpha)
    
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Get 4-hour data for Donchian channel (20 periods)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian upper and lower bands (20-period high/low)
    donch_high = np.full(len(df_4h), np.nan)
    donch_low = np.full(len(df_4h), np.nan)
    for i in range(20, len(df_4h)):
        donch_high[i] = np.max(high_4h[i-20:i])
        donch_low[i] = np.min(low_4h[i-20:i])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Calculate 4-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(4, n):
        if i == 4:
            avg_gain[i] = np.mean(gain[1:5])
            avg_loss[i] = np.mean(loss[1:5])
        else:
            avg_gain[i] = (avg_gain[i-1] * 3 + gain[i]) / 4
            avg_loss[i] = (avg_loss[i-1] * 3 + loss[i]) / 4
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_4 = np.full(n, np.nan)
    rsi_4[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    # Calculate ATR for stop loss (14-period on daily)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    
    # Warmup: need enough data for all indicators
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_34_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(rsi_4[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price above/below daily EMA(34)
        uptrend = price > ema_1d_34_aligned[i]
        downtrend = price < ema_1d_34_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + uptrend + RSI not overbought
            if (price > donch_high_aligned[i] and 
                uptrend and 
                rsi_4[i] < 70):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower + downtrend + RSI not oversold
            elif (price < donch_low_aligned[i] and 
                  downtrend and 
                  rsi_4[i] > 30):
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold or exit
            signals[i] = 0.25
            # Exit: price breaks below Donchian lower OR RSI overbought OR stop loss hit
            if (price < donch_low_aligned[i] or 
                rsi_4[i] > 70 or
                price < entry_price - 2.0 * atr_14_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold or exit
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper OR RSI oversold OR stop loss hit
            if (price > donch_high_aligned[i] or 
                rsi_4[i] < 30 or
                price > entry_price + 2.0 * atr_14_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_DonchianBreakout_EMA34Trend_RSI4_v1"
timeframe = "4h"
leverage = 1.0