#!/usr/bin/env python3
"""
Hypothesis: 1d RSI(2) + 1w Trend Filter with Volume Confirmation.
Long when RSI(2) < 10 (oversold) and 1w EMA50 rising with volume spike.
Short when RSI(2) > 90 (overbought) and 1w EMA50 falling with volume spike.
Exit when RSI(2) crosses above 50 (for longs) or below 50 (for shorts).
This targets mean reversion in daily timeframe with weekly trend filter to avoid counter-trend trades.
Works in both bull and bear markets by following weekly trend direction.
Low trade frequency expected due to strict RSI(2) thresholds.
"""

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
    
    # RSI(2) calculation
    def rsi(close_prices, period=2):
        if len(close_prices) < period + 1:
            return np.full_like(close_prices, np.nan, dtype=float)
        delta = np.diff(close_prices)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing (alpha = 1/period)
        def wilders_smoothing(x, period):
            if len(x) < period:
                return np.full_like(x, np.nan, dtype=float)
            result = np.full_like(x, np.nan, dtype=float)
            result[period-1] = np.mean(x[:period])
            for i in range(period, len(x)):
                result[i] = (result[i-1] * (period-1) + x[i]) / period
            return result
        
        up_smoothed = wilders_smoothing(up, period)
        down_smoothed = wilders_smoothing(down, period)
        
        rs = np.where(down_smoothed != 0, up_smoothed / down_smoothed, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        # Prepend NaN for the first element (since diff reduces length by 1)
        return np.concatenate([[np.nan], rsi_vals])
    
    rsi2 = rsi(close, 2)
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1w close for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi2[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) and 1w EMA50 rising with volume spike
            if (rsi2[i] < 10 and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 (overbought) and 1w EMA50 falling with volume spike
            elif (rsi2[i] > 90 and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI(2) crosses above 50 (for longs) or below 50 (for shorts)
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI(2) >= 50
                if rsi2[i] >= 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI(2) <= 50
                if rsi2[i] <= 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_RSI2_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0