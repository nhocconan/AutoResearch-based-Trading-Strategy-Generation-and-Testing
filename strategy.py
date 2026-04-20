#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period ADX for trend strength (HTF)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    di_plus_1d = wilder_smooth(dm_plus, 14)
    di_minus_1d = wilder_smooth(dm_minus, 14)
    
    # Avoid division by zero
    di_sum = di_plus_1d + di_minus_1d
    dx = np.where(di_sum != 0, 100 * np.abs(di_plus_1d - di_minus_1d) / di_sum, 0)
    adx_1d = wilder_smooth(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 14-period ADX for LTF (same period for comparison)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1_l = high - low
    tr2_l = np.abs(high - np.roll(close, 1))
    tr3_l = np.abs(low - np.roll(close, 1))
    tr_l = np.maximum(tr1_l, np.maximum(tr2_l, tr3_l))
    tr_l[0] = tr1_l[0]
    
    # Directional Movement
    dm_plus_l = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                         np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus_l = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                          np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus_l[0] = 0
    dm_minus_l[0] = 0
    
    atr_l = wilder_smooth(tr_l, 14)
    di_plus_l = wilder_smooth(dm_plus_l, 14)
    di_minus_l = wilder_smooth(dm_minus_l, 14)
    
    di_sum_l = di_plus_l + di_minus_l
    dx_l = np.where(di_sum_l != 0, 100 * np.abs(di_plus_l - di_minus_l) / di_sum_l, 0)
    adx_l = wilder_smooth(dx_l, 14)
    
    # Calculate 14-period RSI for LTF
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def rsi_wilder(data, period):
        result = np.zeros_like(data)
        avg_gain = np.mean(data[:period])
        avg_loss = np.mean(data[:period])
        result[period-1] = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100
        for i in range(period, len(data)):
            avg_gain = (avg_gain * (period - 1) + data[i]) / period
            avg_loss = (avg_loss * (period - 1) + data[i]) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            result[i] = 100 - (100 / (1 + rs))
        return result
    
    rsi_14 = rsi_wilder(gain, 14) - rsi_wilder(loss, 14) + 50  # Simplified RSI calculation
    # Correct RSI calculation
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        adx_htf_val = adx_1d_aligned[i]
        adx_ltf_val = adx_l[i]
        rsi_val = rsi_14[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_htf_val) or np.isnan(adx_ltf_val) or np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong trend (HTF ADX > 25), weakening short-term momentum (LTF ADX < 20), RSI oversold (<30)
            if adx_htf_val > 25 and adx_ltf_val < 20 and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short: Strong trend (HTF ADX > 25), weakening long-term momentum (LTF ADX < 20), RSI overbought (>70)
            elif adx_htf_val > 25 and adx_ltf_val < 20 and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought (>70) or trend weakening (HTF ADX < 20)
            if rsi_val > 70 or adx_htf_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) or trend weakening (HTF ADX < 20)
            if rsi_val < 30 or adx_htf_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_HTF_ADX_LTF_ADX_RSI_MeanReversion
# Uses daily ADX for trend strength filter (HTF ADX > 25)
# Uses 4h ADX for momentum exhaustion (LTF ADX < 20)
# Uses 4h RSI for mean reversion signals (RSI < 30 for long, > 70 for short)
# Session filter: 8-20 UTC to avoid low-volume periods
# Designed for 4h timeframe with ~20-40 trades/year
name = "4h_HTF_ADX_LTF_ADX_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0