#!/usr/bin/env python3
# 4h_ADX_Trend_Reversal_Volume
# Hypothesis: Combines ADX trend strength with RSI mean reversion and volume confirmation.
# Long when ADX > 25 (trending) and RSI < 30 (oversold) with volume > 1.5x average.
# Short when ADX > 25 and RSI > 70 (overbought) with volume confirmation.
# Uses 1d timeframe for ADX/RSI to avoid noise, 4h for execution.
# Designed for 20-40 trades/year with strong trend and value conditions to avoid false signals.

name = "4h_ADX_Trend_Reversal_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1d data for ADX and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components on 1d data
    # True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 
                                   np.abs(high_1d[0] - close_1d[0]), 
                                   np.abs(low_1d[0] - close_1d[0])])], tr1])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Handle first element
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero
    dx = np.where((di_plus + di_minus) != 0, dx, 0)
    
    # ADX is smoothed DX
    adx = wilder_smooth(dx, period)
    
    # Calculate RSI on 1d data
    delta = np.diff(close_1d)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def rsi_wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = rsi_wilder_smooth(gain, 14)
    avg_loss = rsi_wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Strong trend (ADX > 25) + oversold (RSI < 30) + volume confirmation
            if adx_val > 25 and rsi_val < 30 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Strong trend (ADX > 25) + overbought (RSI > 70) + volume confirmation
            elif adx_val > 25 and rsi_val > 70 and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakening (ADX < 20) or overbought (RSI > 70)
            if adx_val < 20 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakening (ADX < 20) or oversold (RSI < 30)
            if adx_val < 20 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals