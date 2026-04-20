#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX (14) for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_series(x, period):
        result = np.full_like(x, np.nan, dtype=float)
        if len(x) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(x[0:period])
        # Wilder's smoothing
        for i in range(period, len(x)):
            result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr_1w = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = smooth_series(dx, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 1d RSI (14) for mean reversion signals
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def rsi_smooth(x, period):
        result = np.full_like(x, np.nan, dtype=float)
        if len(x) < period:
            return result
        avg_gain = np.mean(gain[0:period])
        avg_loss = np.mean(loss[0:period])
        for i in range(period, len(x)):
            avg_gain = (avg_gain * (period-1) + gain[i]) / period
            avg_loss = (avg_loss * (period-1) + loss[i]) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            result[i] = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100
        return result
    
    rsi_1d = rsi_smooth(gain, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 12h ATR (14) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_1d_aligned[i]
        adx_val = adx_1w_aligned[i]
        atr_12h_val = atr_12h[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(adx_val) or 
            np.isnan(atr_12h_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold in weak trend (ADX < 25) - mean reversion opportunity
            if rsi_val < 30 and adx_val < 25:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in weak trend (ADX < 25) - mean reversion opportunity
            elif rsi_val > 70 and adx_val < 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or stoploss hit
            if rsi_val > 70 or close_val < prices['high'].iloc[i] - 1.5 * atr_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or stoploss hit
            if rsi_val < 30 or close_val > prices['low'].iloc[i] + 1.5 * atr_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_1wADX_1dRSI_MeanReversion_V1
# Uses 1-week ADX to filter for weak trends (ADX < 25) where mean reversion works
# Enters long when 1-day RSI < 30 (oversold) in weak trend
# Enters short when 1-day RSI > 70 (overbought) in weak trend
# Uses 12h ATR for 1.5x ATR stoploss
# Works in both bull and bear markets by fading extremes in low-volatility regimes
# Designed for 12h timeframe with ~12-37 trades/year
name = "12h_1wADX_1dRSI_MeanReversion_V1"
timeframe = "12h"
leverage = 1.0