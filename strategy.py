#!/usr/bin/env python3
"""
4h_1d_1w_Keltner_Adaptive_Momentum_v1
Concept: Adaptive Keltner channel with momentum confirmation for trend capture in all regimes.
- Long: Close > upper_Keltner(ATR multiplier adapts to volatility regime) AND RSI > 50
- Short: Close < lower_Keltner AND RSI < 50
- Exit: Opposite Keltner band touch (long exits at lower band, short at upper)
- Uses 1d trend filter (price > 200 EMA) and 1w volatility regime for adaptive bands
- Position sizing: 0.28 (balanced for risk/return)
- Target: ~120 total trades over 4 years to avoid fee drag
- Works in bull/bear: Adaptive bands prevent whipsaw in high vol, trend filter avoids counter-trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Keltner_Adaptive_Momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d: 200 EMA for trend filter ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 1w: ATR for volatility regime (adaptive multiplier) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(20) on 1w
    atr_20_1w = pd.Series(tr_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volatility regime: normalize ATR by price
    atr_norm_1w = atr_20_1w / close_1w
    atr_norm_ma_1w = pd.Series(atr_norm_1w).rolling(window=10, min_periods=10).mean().values
    
    # Adaptive ATR multiplier: higher in high vol, lower in low vol
    # Base multiplier 1.5, scales with volatility percentile
    atr_multiplier_base = 1.5
    atr_percentile = pd.Series(atr_norm_1w).rolling(window=50, min_periods=20).rank(pct=True).values
    atr_multiplier = atr_multiplier_base * (0.5 + atr_percentile)  # ranges from 0.75x to 2.25x base
    
    atr_multiplier_aligned = align_htf_to_ltf(prices, df_1w, atr_multiplier)
    
    # === 4h: Price, ATR, RSI for Keltner channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical Price for Keltner
    tp = (high + low + close) / 3.0
    
    # ATR(10) for 4h
    tr1_4h = np.abs(high[1:] - low[1:])
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.max([tr1_4h[0], tr2_4h[0], tr3_4h[0]])], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_10_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # EMA(20) of Typical Price for middle band
    ema_tp_20 = pd.Series(tp).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Adaptive Keltner bands
    upper_keltner = ema_tp_20 + (atr_10_4h * atr_multiplier_aligned)
    lower_keltner = ema_tp_20 - (atr_10_4h * atr_multiplier_aligned)
    
    # RSI(14) for momentum confirmation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Prepend first value as 50 (neutral)
    rsi = np.concatenate([[50.0], rsi])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        upper_kelt = upper_keltner[i]
        lower_kelt = lower_keltner[i]
        ema_200 = ema_200_1d_aligned[i]
        rsi_val = rsi[i]
        current_close = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_kelt) or np.isnan(lower_kelt) or 
            np.isnan(ema_200) or np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price > 200 EMA for long bias, < for short bias
        long_bias = current_close > ema_200
        short_bias = current_close < ema_200
        
        if position == 0:
            # Long: price above upper Keltner with bullish momentum and trend
            if current_close > upper_kelt and rsi_val > 50 and long_bias:
                signals[i] = 0.28
                position = 1
            # Short: price below lower Keltner with bearish momentum and trend
            elif current_close < lower_kelt and rsi_val < 50 and short_bias:
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Long exit: price touches or crosses lower Keltner (mean reversion)
            if current_close < lower_kelt:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Short exit: price touches or crosses upper Keltner (mean reversion)
            if current_close > upper_kelt:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals