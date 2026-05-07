#!/usr/bin/env python3
name = "4h_WMA_Pullback_Trend_Confirm"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter and ATR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly WMA(21) for trend filter
    weights = np.arange(1, 22)
    wma_21_1w = np.convolve(df_1w['close'].values, weights, mode='valid')
    wma_21_1w = np.concatenate([np.full(len(df_1w) - len(wma_21_1w), np.nan), wma_21_1w])
    
    # Weekly ATR(14) for volatility filter
    tr1 = df_1w['high'].values - df_1w['low'].values
    tr2 = np.abs(df_1w['high'].values - np.concatenate([df_1w['close'].values[:-1], [np.nan]]))
    tr3 = np.abs(df_1w['low'].values - np.concatenate([df_1w['close'].values[:-1], [np.nan]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly indicators to 4h timeframe
    wma_21_4h = align_htf_to_ltf(prices, df_1w, wma_21_1w)
    atr_14_4h = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # 4h EMA(50) for dynamic support/resistance
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(wma_21_4h[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(ema_50_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend: price above/below WMA21
        weekly_uptrend = close[i] > wma_21_4h[i]
        weekly_downtrend = close[i] < wma_21_4h[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_4h[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: pullback to EMA50 in weekly uptrend with sufficient volatility
            if weekly_uptrend and close[i] > ema_50_4h[i] * 0.998 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: pullback to EMA50 in weekly downtrend with sufficient volatility
            elif weekly_downtrend and close[i] < ema_50_4h[i] * 1.002 and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: weekly trend reversal or price breaks below EMA50
            if not weekly_uptrend or close[i] < ema_50_4h[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: weekly trend reversal or price breaks above EMA50
            if not weekly_downtrend or close[i] > ema_50_4h[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly trend + 4h EMA pullback strategy
# - Uses weekly WMA(21) as primary trend filter (superior to EMA in trending markets)
# - Enters on pullbacks to 4h EMA(50) in direction of weekly trend
# - Volatility filter (ATR > 1% of price) prevents choppy market entries
# - Works in both bull (buy pullbacks in uptrend) and bear (sell pullbacks in downtrend)
# - Position size 0.25 targets ~20-40 trades/year to minimize fee drag
# - Weekly timeframe for structure, 4h for precise entry timing
# - Simple, robust logic with minimal parameters to avoid overfitting