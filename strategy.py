#!/usr/bin/env python3
# 1h_4h1d_Trend_Filtered_Momentum_v2
# Hypothesis: Use 4h trend (EMA50) and 1d volatility (ATR) as filters for 1h momentum entries.
# Only trade during active sessions (08-20 UTC) to avoid low-liquidity noise.
# Long: price > 4h EMA50 AND 1h close > 1h open AND ATR ratio > 0.8
# Short: price < 4h EMA50 AND 1h close < 1h open AND ATR ratio > 0.8
# Position size fixed at 0.20 to manage drawdown. Target 15-35 trades/year.

name = "1h_4h1d_Trend_Filtered_Momentum_v2"
timeframe = "1h"
leverage = 1.0

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
    open_ = prices['open'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[0:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = (ema_50_4h[i-1] * 49 + close_4h[i]) / 50
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr = np.maximum(high_1d - low_1d, np.maximum(abs(high_1d - np.roll(close_1d, 1)), abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # first TR
    atr_14_1d = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.mean(tr[0:14])
        for i in range(14, len(tr)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1h ATR(14) for volatility ratio
    tr_1h = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr_1h[0] = high[0] - low[0]
    atr_14_1h = np.full_like(tr_1h, np.nan)
    if len(tr_1h) >= 14:
        atr_14_1h[13] = np.mean(tr_1h[0:14])
        for i in range(14, len(tr_1h)):
            atr_14_1h[i] = (atr_14_1h[i-1] * 13 + tr_1h[i]) / 14
    atr_ratio = np.full_like(close, np.nan)
    valid = (~np.isnan(atr_14_1h)) & (~np.isnan(atr_14_1d_aligned)) & (atr_14_1d_aligned != 0)
    atr_ratio[valid] = atr_14_1h[valid] / atr_14_1d_aligned[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # Ensure EMA and ATR are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_ratio[i]) or
            not (8 <= hours[i] <= 20)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: uptrend + bullish candle + sufficient volatility
            if (close[i] > ema_50_4h_aligned[i] and 
                close[i] > open_[i] and 
                atr_ratio[i] > 0.8):
                signals[i] = 0.20
                position = 1
            # Enter short: downtrend + bearish candle + sufficient volatility
            elif (close[i] < ema_50_4h_aligned[i] and 
                  close[i] < open_[i] and 
                  atr_ratio[i] > 0.8):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or bearish candle
            if close[i] < ema_50_4h_aligned[i] or close[i] < open_[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend reversal or bullish candle
            if close[i] > ema_50_4h_aligned[i] or close[i] > open_[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals