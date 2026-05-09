#!/usr/bin/env python3
# 6h_ADX_DI_Crossover_1dTrendFilter
# Strategy: Use ADX and DI crossover on 6h for trend strength and direction, filtered by 1d EMA trend.
# Long when +DI crosses above -DI, ADX > 25, and price above 1d EMA(50).
# Short when -DI crosses above +DI, ADX > 25, and price below 1d EMA(50).
# Exit when ADX falls below 20 (weakening trend) or DI crossover reverses.
# Designed for 6h timeframe to capture strong trends with low trade frequency.

name = "6h_ADX_DI_Crossover_1dTrendFilter"
timeframe = "6h"
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
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ADX and DI (14-period)
    def calculate_adx_di(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.append(close[0], close[:-1]))
        tr3 = np.abs(low - np.append(close[0], close[:-1]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - np.append(high[0], high[:-1])
        down_move = np.append(low[0], low[:-1]) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            for i in range(len(data)):
                if i == 0:
                    result[i] = data[i]
                elif np.isnan(result[i-1]):
                    result[i] = data[i]
                else:
                    result[i] = (1 - alpha) * result[i-1] + alpha * data[i]
            # Set first 'period' values to NaN to match standard ADX
            result[:period-1] = np.nan
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)
        
        # Calculate DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, period)
        
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx_di(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: +DI crosses above -DI, ADX > 25, price above 1d EMA50
            if (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1] and 
                adx[i] > 25 and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: -DI crosses above +DI, ADX > 25, price below 1d EMA50
            elif (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1] and 
                  adx[i] > 25 and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: ADX < 20 or -DI crosses above +DI
            if adx[i] < 20 or (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: ADX < 20 or +DI crosses above -DI
            if adx[i] < 20 or (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals