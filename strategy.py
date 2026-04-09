#!/usr/bin/env python3
# 6h_1d_adx_trend_v1
# Hypothesis: 6-hour trend following using 1-day ADX for trend strength and 6-hour price action for entry.
# Long when ADX(14) > 25 (strong trend) and price crosses above 6h EMA(21) with bullish engulfing candle.
# Short when ADX(14) > 25 and price crosses below 6h EMA(21) with bearish engulfing candle.
# Exit when ADX drops below 20 (weakening trend) or opposite signal occurs.
# Works in bull markets via trend continuation and in bear markets via short trends.
# Uses discrete position sizes (0.25) to limit turnover and control drawdown.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_adx_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period])  # Skip index 0 (nan)
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            else:
                result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA(21)
    ema_21 = np.full(n, np.nan)
    if n >= 21:
        multiplier = 2 / (21 + 1)
        ema_21[20] = np.mean(open_price[0:21])  # Simple average of first 21 opens
        for i in range(21, n):
            ema_21[i] = (open_price[i] - ema_21[i-1]) * multiplier + ema_21[i-1]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):  # Start after EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(ema_21[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ADX weakens (<20) or bearish engulfing forms
            if (adx_aligned[i] < 20 or 
                (close[i] < open_price[i] and 
                 open_price[i-1] < close[i-1] and  # Previous bullish
                 close[i] <= open_price[i-1] and   # Current close below prev open
                 open_price[i] >= close[i-1])):    # Current open above prev close
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX weakens (<20) or bullish engulfing forms
            if (adx_aligned[i] < 20 or 
                (close[i] > open_price[i] and 
                 open_price[i-1] > close[i-1] and  # Previous bearish
                 close[i] >= open_price[i-1] and   # Current close above prev open
                 open_price[i] <= close[i-1])):    # Current open below prev close
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: strong trend (ADX>25) and bullish engulfing above EMA
            if (adx_aligned[i] > 25 and 
                close[i] > ema_21[i] and
                close[i] > open_price[i] and  # Bullish candle
                open_price[i-1] < close[i-1] and  # Previous bullish
                close[i] <= open_price[i-1] and   # Current close below prev open
                open_price[i] >= close[i-1]):    # Current open above prev close (engulfing)
                position = 1
                signals[i] = 0.25
            # Enter short: strong trend (ADX>25) and bearish engulfing below EMA
            elif (adx_aligned[i] > 25 and 
                  close[i] < ema_21[i] and
                  close[i] < open_price[i] and  # Bearish candle
                  open_price[i-1] > close[i-1] and  # Previous bearish
                  close[i] >= open_price[i-1] and   # Current close above prev open
                  open_price[i] <= close[i-1]):    # Current open below prev close (engulfing)
                position = -1
                signals[i] = -0.25
    
    return signals