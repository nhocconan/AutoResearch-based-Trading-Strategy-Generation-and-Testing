#!/usr/bin/env python3
"""
4h_1D_Momentum_Confirmation_V1
4-hour strategy using 1-day momentum (ROC-10) and volume confirmation.
Enters long when 1-day ROC > 0 and 4h price > 20-period EMA with volume above average.
Enters short when 1-day ROC < 0 and 4h price < 20-period EMA with volume above average.
Uses ROC for trend direction, EMA for dynamic support/resistance, and volume for confirmation.
Designed to work in both bull and bear markets by following higher timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1-day ROC for trend direction ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate ROC-10 on daily close
    roc_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    for i in range(10, len(close_1d)):
        if close_1d[i-10] != 0:
            roc_1d[i] = (close_1d[i] - close_1d[i-10]) / close_1d[i-10] * 100.0
    
    # Align ROC to 4h timeframe
    roc_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_1d)
    
    # === 4h EMA(20) for dynamic support/resistance ===
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 4h Volume for confirmation ===
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(roc_1d_aligned[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 4h volume for confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend direction from 1-day ROC
        bullish_trend = roc_1d_aligned[i] > 0
        bearish_trend = roc_1d_aligned[i] < 0
        
        # Price position relative to EMA
        price_above_ema = close[i] > ema_20[i]
        price_below_ema = close[i] < ema_20[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish 1-day trend + price above EMA + volume confirmation
            if bullish_trend and price_above_ema and vol_confirmed:
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish 1-day trend + price below EMA + volume confirmation
            elif bearish_trend and price_below_ema and vol_confirmed:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse when trend changes or price crosses EMA
        elif position == 1:
            # Exit long: bearish trend or price below EMA
            if bearish_trend or price_below_ema:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price above EMA
            if bullish_trend or price_above_ema:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1D_Momentum_Confirmation_V1"
timeframe = "4h"
leverage = 1.0