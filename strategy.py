#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Filter_v1
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets. 
Enter long when price > KAMA AND RSI(14) > 55, short when price < KAMA AND RSI(14) < 45.
Use 1d ADX > 25 to filter for trending markets only, reducing whipsaws in ranging periods.
Requires volume > 1.3x 20-period average for confirmation.
Target: 20-40 trades/year by combining trend, momentum, regime, and volume filters.
Works in bull via trend following and in bear via short signals during strong trends.
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
    
    # Get 1d data for ADX (trend strength filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) - measures trend strength
    adx_period = 14
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+,
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            result[period-1] = np.nansum(x[1:period])  # skip first NaN
            for i in range(period, len(x)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    tr_smooth = wilders_smoothing(tr, adx_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, adx_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, adx_period)
    
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, adx_period)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # KAMA (Adaptive Moving Average) on 4h close
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Efficiency Ratio
    er_period = 10
    change = np.abs(np.concatenate([[np.nan], close_4h[1:] - close_4h[:-1]]))
    travel = np.abs(np.concatenate([[np.nan], close_4h[er_period:] - close_4h[:-er_period]]))
    travel = np.concatenate([np.full(er_period, np.nan), travel])  # align
    
    er = np.where(travel != 0, change / travel, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) > 0:
        kama_4h[0] = close_4h[0]
        for i in range(1, len(close_4h)):
            if not np.isnan(sc[i]):
                kama_4h[i] = kama_4h[i-1] + sc[i] * (close_4h[i] - kama_4h[i-1])
            else:
                kama_4h[i] = kama_4h[i-1]
    
    # Align KAMA to 4h timeframe (already aligned via index)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # RSI(14) on 4h
    rsi_period = 14
    delta = np.diff(close_4h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_4h, np.nan)
    avg_loss = np.full_like(close_4h, np.nan)
    
    if len(close_4h) >= rsi_period + 1:
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        for i in range(rsi_period + 1, len(close_4h)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_4h = 100 - (100 / (1 + rs))
    else:
        rsi_4h = np.full_like(close_4h, np.nan)
    
    # Align RSI to 4h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Volume confirmation: volume > 1.3x 20-period average on 4h
    vol_4h = df_4h['volume'].values
    vol_period = 20
    vol_ma_4h = np.full_like(vol_4h, np.nan)
    
    if len(vol_4h) >= vol_period:
        for i in range(vol_period, len(vol_4h)):
            vol_ma_4h[i] = np.mean(vol_4h[i - vol_period:i])
    
    # Align volume MA to 4h timeframe
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(adx_period, er_period, rsi_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(kama_4h_aligned[i]) or 
            np.isnan(rsi_4h_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 (trending market)
        trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = vol_4h[i] > 1.3 * vol_ma_4h_aligned[i]
        
        if position == 0 and trending:
            # Long: price > KAMA AND RSI > 55 AND volume
            if close[i] > kama_4h_aligned[i] and rsi_4h_aligned[i] > 55 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA AND RSI < 45 AND volume
            elif close[i] < kama_4h_aligned[i] and rsi_4h_aligned[i] < 45 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA OR RSI < 40
            if close[i] < kama_4h_aligned[i] or rsi_4h_aligned[i] < 40:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA OR RSI > 60
            if close[i] > kama_4h_aligned[i] or rsi_4h_aligned[i] > 60:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_With_RSI_Filter_v1"
timeframe = "4h"
leverage = 1.0