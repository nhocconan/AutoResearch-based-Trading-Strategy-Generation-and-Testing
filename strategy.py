#!/usr/bin/env python3
"""
1h_4d_Hybrid_Momentum_Strategy
Hypothesis: Combines 4h trend direction with 1h momentum entries and 1d regime filter.
4h EMA21 defines trend, 1h RSI < 30 (oversold) or > 70 (overbought) with volume confirmation for entries.
1d ADX > 25 filters for trending markets only. Works in bull (buy pullbacks in uptrend) and bear (sell bounces in downtrend).
Target: 20-30 trades/year per symbol. Uses 4h/1d for direction/regime, 1h for timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA21 for trend direction
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d ADX for regime filter (trending market)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    )
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    period = 14
    if len(tr) >= period:
        atr_1d = wilder_smooth(tr, period)
        plus_di_1d = 100 * wilder_smooth(plus_dm, period) / atr_1d
        minus_di_1d = 100 * wilder_smooth(minus_dm, period) / atr_1d
        dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
        adx_1d = wilder_smooth(dx_1d, period)
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, np.nan)
    
    # 1h RSI for momentum entries
    def rsi(arr, period=14):
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(arr)
        avg_loss = np.zeros_like(arr)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_1h = rsi(close, 14)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(rsi_1h[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (ADX > 25)
        if adx_1d_aligned[i] <= 25:
            # No trend - stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Long: uptrend (price > EMA21) + RSI oversold (<30) + volume expansion
        long_condition = (close[i] > ema_21_4h_aligned[i]) and (rsi_1h[i] < 30) and volume_expansion[i]
        
        # Short: downtrend (price < EMA21) + RSI overbought (>70) + volume expansion
        short_condition = (close[i] < ema_21_4h_aligned[i]) and (rsi_1h[i] > 70) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4d_Hybrid_Momentum_Strategy"
timeframe = "1h"
leverage = 1.0