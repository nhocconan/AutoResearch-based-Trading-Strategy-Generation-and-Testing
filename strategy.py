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
    
    # Get 12h data for trend and volatility
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA50 for trend
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema50_12h[i] = (close_12h[i] * 2 + ema50_12h[i-1] * 49) / 50
    
    # Calculate 12h ATR(14) for volatility filter
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[high_12h[0] - low_12h[0]], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = np.full(len(close_12h), np.nan)
    for i in range(len(close_12h)):
        if i < 14:
            atr_12h[i] = np.mean(tr_12h[:i+1]) if i > 0 else tr_12h[0]
        else:
            atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Align 12h indicators to 6h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 6h ATR(14) for entry conditions
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 6h volume MA(20)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 12h EMA50 (50), 12h ATR (14), 6h ATR (14), volume MA (20)
    start_idx = max(50, 14, 14, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_12h_aligned[i]) or
            np.isnan(atr_12h_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2.0x average volume
        volume_confirmation = vol_ratio > 2.0
        
        # Volatility filter: only trade when 12h ATR is above its 50-period average
        if i >= 50 + 12:  # Need enough history for 12h ATR average
            atr_12h_avg = np.mean(atr_12h_aligned[max(0, i-50):i+1])
            vol_filter = atr_12h_aligned[i] > atr_12h_avg * 0.6
        else:
            vol_filter = True  # No filter during warmup
        
        # Trend filter: price vs 12h EMA50
        uptrend = price > ema50_12h_aligned[i]
        downtrend = price < ema50_12h_aligned[i]
        
        if position == 0:
            # Long: uptrend + volatility + volume + break above ATR-based resistance
            if uptrend and vol_filter and volume_confirmation:
                resistance = ema50_12h_aligned[i] + 1.5 * atr[i]
                if price > resistance:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            # Short: downtrend + volatility + volume + break below ATR-based support
            elif downtrend and vol_filter and volume_confirmation:
                support = ema50_12h_aligned[i] - 1.5 * atr[i]
                if price < support:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: trend reversal or volatility drop
            if not uptrend or atr_12h_aligned[i] < np.mean(atr_12h_aligned[max(0, i-50):i+1]) * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: trend reversal or volatility drop
            if not downtrend or atr_12h_aligned[i] < np.mean(atr_12h_aligned[max(0, i-50):i+1]) * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "6h_12h_EMA50_ATR_Volume_Breakout_v1"
timeframe = "6h"
leverage = 1.0