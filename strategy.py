#!/usr/bin/env python3
# 4h_Supertrend_RSI_Combo
# Hypothesis: In bull markets, Supertrend captures strong trends; in bear markets, RSI extremes with trend filter capture mean reversion.
# Combines Supertrend for trend direction and RSI for entry timing, with volume confirmation to filter false signals.
# Designed for low trade frequency (20-40/year) to minimize fee drag while capturing both trending and ranging markets.

name = "4h_Supertrend_RSI_Combo"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Supertrend calculation
    atr_period = 10
    atr_multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # No previous close for first bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    def rma(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            res[period-1] = np.mean(arr[:period])
            # Subsequent values using Wilder's smoothing
            for i in range(period, len(arr)):
                res[i] = (arr[i] + (period-1) * res[i-1]) / period
        return res
    
    atr = rma(tr, atr_period)
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + atr_multiplier * atr
    basic_lb = (high + low) / 2 - atr_multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.full_like(close, np.nan)
    final_lb = np.full_like(close, np.nan)
    
    for i in range(len(close)):
        if np.isnan(basic_ub[i]) or np.isnan(basic_lb[i]):
            continue
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if close[i-1] <= final_ub[i-1]:
                final_ub[i] = min(basic_ub[i], final_ub[i-1])
            else:
                final_ub[i] = basic_ub[i]
                
            if close[i-1] >= final_lb[i-1]:
                final_lb[i] = max(basic_lb[i], final_lb[i-1])
            else:
                final_lb[i] = basic_lb[i]
    
    # Supertrend
    supertrend = np.full_like(close, np.nan)
    for i in range(len(close)):
        if np.isnan(final_ub[i]) or np.isnan(final_lb[i]):
            continue
        if i == 0:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1]:
                if close[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                else:
                    supertrend[i] = final_lb[i]
            else:
                if close[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                else:
                    supertrend[i] = final_ub[i]
    
    # Supertrend trend direction (1 = uptrend, -1 = downtrend)
    trend = np.where(close > supertrend, 1, -1)
    
    # RSI calculation
    def rsi(arr, period=14):
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing for average gain and loss
        avg_gain = rma(gain, period)
        avg_loss = rma(loss, period)
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_vals = rsi(close, 14)
    
    # Get 1d data for volume average (longer-term volume context)
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = rma(vol_1d, 20)  # 20-day volume average
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(supertrend[i]) or np.isnan(rsi_vals[i]) or \
           np.isnan(trend[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-day average
        volume_confirm = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.3 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: uptrend + RSI oversold + volume confirmation
            if trend[i] == 1 and rsi_vals[i] < 30 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + RSI overbought + volume confirmation
            elif trend[i] == -1 and rsi_vals[i] > 70 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend change or RSI overbought
            if trend[i] == -1 or rsi_vals[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend change or RSI oversold
            if trend[i] == 1 or rsi_vals[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals