#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and 1d ADX regime filter.
# Long when price breaks above R3 (bullish breakout) AND price > 4h EMA50 AND 1d ADX > 25 (trending market).
# Short when price breaks below S3 (bearish breakout) AND price < 4h EMA50 AND 1d ADX > 25.
# Exits when price returns to the Camarilla pivot point (mean reversion to equilibrium) OR ADX < 20 (regime shift to ranging).
# Uses discrete position sizing (0.20) to limit fee churn. Designed for BTC/ETH robustness by capturing breakouts in trending markets while avoiding false breakouts in ranging markets via ADX filter.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dADX_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ADX for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(low_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(tr) < period:
        return np.zeros(n)
        
    atr = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    divisor = np.where(atr == 0, 1, atr)
    plus_di = 100 * plus_dm_smooth / divisor
    minus_di = 100 * minus_dm_smooth / divisor
    
    dx = np.where((plus_di + minus_di) == 0, 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = wilder_smooth(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla pivot levels from previous day
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Need previous day's OHLC for Camarilla calculation
        if i < 24:  # need at least 24 hours of 1h data for previous day
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC (24 hours ago in 1h timeframe)
        prev_high = high[i-24]
        prev_low = low[i-24]
        prev_close = close[i-24]
        
        # Calculate Camarilla levels
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        r3 = pivot + (range_val * 1.1 / 4)
        s3 = pivot - (range_val * 1.1 / 4)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND price > 4h EMA50 AND ADX > 25 (trending)
            if (close[i] > r3 and 
                close[i] > ema_50_4h_aligned[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 AND price < 4h EMA50 AND ADX > 25 (trending)
            elif (close[i] < s3 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot (mean reversion) OR ADX < 20 (ranging)
            if (close[i] <= pivot or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to pivot (mean reversion) OR ADX < 20 (ranging)
            if (close[i] >= pivot or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals