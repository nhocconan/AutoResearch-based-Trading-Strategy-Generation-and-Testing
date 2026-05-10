#!/usr/bin/env python3
# 6h_12h_Adaptive_Keltner_Breakout_Trend
# Hypothesis: 6h breakout from adaptive Keltner channels (ATR-based) with 12h trend filter and volume confirmation.
# Uses ATR multiplier that adapts to volatility regime (lower multiplier in high vol = tighter bands).
# Long: breakout above upper band in uptrend with volume spike. Short: breakdown below lower band in downtrend with volume spike.
# Designed for 6h timeframe targeting 12-30 trades/year per symbol. Works in bull/bear by requiring trend alignment.
# Keltner channels adapt to volatility, reducing whipsaws in ranging markets and capturing trends effectively.

name = "6h_12h_Adaptive_Keltner_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend and volatility calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h ATR for volatility regime
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate adaptive multiplier based on ATR percentile (20-period)
    atr_ma = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_12h / atr_ma  # Current ATR vs average ATR
    # Adaptive multiplier: 1.5 in low vol, 2.5 in high vol (clipped)
    multiplier = 1.5 + np.clip((atr_ratio - 1.0) * 2.0, 0.0, 1.0)  # Range 1.5 to 2.5
    
    # Calculate 12h EMA for trend (middle of Keltner)
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner channels
    upper_12h = ema_12h + multiplier * atr_12h
    lower_12h = ema_12h - multiplier * atr_12h
    
    # Align 12h indicators to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Volume confirmation (20-period for 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for ATR calculation + EMA + vol MA
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_12h_aligned[i]) or
            np.isnan(upper_12h_aligned[i]) or
            np.isnan(lower_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 12h close > EMA
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        uptrend = close_12h_aligned[i] > ema_12h_aligned[i]
        downtrend = close_12h_aligned[i] < ema_12h_aligned[i]
        
        # Volume confirmation (2x average for significance)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above upper Keltner band in uptrend with volume spike
            if close[i] > upper_12h_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower Keltner band in downtrend with volume spike
            elif close[i] < lower_12h_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close below EMA (middle band) or trend fails
                if close[i] < ema_12h_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close above EMA (middle band) or trend fails
                if close[i] > ema_12h_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals