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
    
    # Get daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.zeros(len(df_1d))
    for i in range(len(tr_1d)):
        if i < 13:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate ATR ratio: current ATR / 20-period average ATR (volatility regime filter)
    atr_ma_20 = np.full(len(atr_1d), np.nan)
    for i in range(20, len(atr_1d)):
        atr_ma_20[i] = np.mean(atr_1d[i-20:i])
    
    atr_ratio = np.full(len(atr_1d), np.nan)
    for i in range(len(atr_1d)):
        if not np.isnan(atr_ma_20[i]) and atr_ma_20[i] > 0:
            atr_ratio[i] = atr_1d[i] / atr_ma_20[i]
    
    # Align ATR ratio to 4h timeframe (volatility regime filter)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Get 4h data for Donchian channel (entry signal)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channel (20-period)
    donchian_high = np.full(len(df_4h), np.nan)
    donchian_low = np.full(len(df_4h), np.nan)
    
    for i in range(len(df_4h)):
        if i >= 19:
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
        elif i >= 0:
            donchian_high[i] = np.max(high_4h[0:i+1])
            donchian_low[i] = np.min(low_4h[0:i+1])
    
    # Align Donchian channels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 4h close for trend filter
    close_4h = df_4h['close'].values
    # Calculate EMA(50) for trend filter
    ema_4h_50 = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        alpha = 2 / (50 + 1)
        for i in range(len(close_4h)):
            if i < 49:
                ema_4h_50[i] = np.mean(close_4h[0:i+1]) if i > 0 else close_4h[i]
            else:
                if np.isnan(ema_4h_50[i-1]):
                    ema_4h_50[i] = np.mean(close_4h[i-49:i+1])
                else:
                    ema_4h_50[i] = close_4h[i] * alpha + ema_4h_50[i-1] * (1 - alpha)
    
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Calculate 4h ATR(14) for stop loss
    tr1_4h = high_4h[1:] - low_4h[1:]
    tr2_4h = np.abs(high_4h[1:] - close_4h[:-1])
    tr3_4h = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[high_4h[0] - low_4h[0]], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    
    atr_4h = np.zeros(len(df_4h))
    for i in range(len(tr_4h)):
        if i < 13:
            atr_4h[i] = np.mean(tr_4h[:i+1]) if i > 0 else tr_4h[i]
        else:
            atr_4h[i] = (atr_4h[i-1] * 13 + tr_4h[i]) / 14
    
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(50, 20)  # EMA50 needs 50, Donchian needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_4h_50_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = 1.0  # Volume confirmation removed to reduce trades
        
        # Volatility regime filter: only trade in normal to high volatility (ATR ratio > 0.8)
        volatility_filter = atr_ratio_aligned[i] > 0.8
        
        if position == 0:
            # Long: price breaks above Donchian high with uptrend and volatility filter
            if (volatility_filter and 
                price > donchian_high_aligned[i] and 
                close[i-1] <= donchian_high_aligned[i] and  # just broke out
                ema_4h_50_aligned[i] > ema_4h_50_aligned[i-1]):  # uptrend
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with downtrend and volatility filter
            elif (volatility_filter and 
                  price < donchian_low_aligned[i] and 
                  close[i-1] >= donchian_low_aligned[i] and  # just broke down
                  ema_4h_50_aligned[i] < ema_4h_50_aligned[i-1]):  # downtrend
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches Donchian low or trend turns down
            if (price <= donchian_low_aligned[i] or 
                ema_4h_50_aligned[i] < ema_4h_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price reaches Donchian high or trend turns up
            if (price >= donchian_high_aligned[i] or 
                ema_4h_50_aligned[i] > ema_4h_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_DonchianBreakout_EMA50_Trend_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0