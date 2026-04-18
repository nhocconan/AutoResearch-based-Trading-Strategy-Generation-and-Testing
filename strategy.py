#!/usr/bin/env python3
"""
1h_4h_1d_ADX_Momentum_v1
Hypothesis: Use 4h ADX > 25 to identify trending conditions, 1d EMA(50) for long-term trend direction, and 1h RSI(2) for precise entry timing. This combination filters out choppy markets, captures momentum in both bull and bear regimes, and limits trades to 15-30/year by requiring multi-timeframe alignment. Works in bull markets via long entries when price > 1d EMA50 and RSI(2) < 10, and in bear markets via short entries when price < 1d EMA50 and RSI(2) > 90, only when 4h ADX confirms trend strength.
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
    
    # Get 4h data for ADX (trend strength filter)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value: simple average
            result[period-1] = np.nansum(arr[1:period]) 
            # Wilder smoothing
            for i in range(period, len(arr)):
                if np.isnan(result[i-1]):
                    result[i] = np.nan
                else:
                    result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result
        
        atr = smooth_wilder(tr, period)
        dm_plus_smooth = smooth_wilder(dm_plus, period)
        dm_minus_smooth = smooth_wilder(dm_minus, period)
        
        # DI+ and DI-
        di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) > 0, 
                      100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = smooth_wilder(dx, period)
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Get 1d data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1h RSI(2) for entry timing
    def calculate_rsi(close, period=2):
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder smoothing
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            result[period-1] = np.nansum(arr[1:period])
            for i in range(period, len(arr)):
                if np.isnan(result[i-1]):
                    result[i] = np.nan
                else:
                    result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result
        
        avg_gain = smooth_wilder(gain, period)
        avg_loss = smooth_wilder(loss, period)
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_2 = calculate_rsi(close, 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need EMA50 and RSI2 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_2[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: ADX > 25 (trending), price > 1d EMA50 (bullish bias), RSI(2) < 10 (oversold)
            if (adx_4h_aligned[i] > 25 and 
                close[i] > ema_50_1d_aligned[i] and 
                rsi_2[i] < 10):
                signals[i] = 0.20
                position = 1
            # Short entry: ADX > 25 (trending), price < 1d EMA50 (bearish bias), RSI(2) > 90 (overbought)
            elif (adx_4h_aligned[i] > 25 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  rsi_2[i] > 90):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: RSI(2) > 80 (overbought) or trend weakness (ADX < 20)
            if (rsi_2[i] > 80 or adx_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI(2) < 20 (oversold) or trend weakness (ADX < 20)
            if (rsi_2[i] < 20 or adx_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_ADX_Momentum_v1"
timeframe = "1h"
leverage = 1.0