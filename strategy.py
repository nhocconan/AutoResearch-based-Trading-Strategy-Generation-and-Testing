#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 12h EMA50 trend filter and 1d ADX regime filter. 
# Elder Ray measures bull/bear strength relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13. 
# Long when Bull Power > 0 and rising (making higher low) AND price > 12h EMA50 AND 1d ADX > 25 (trending market). 
# Short when Bear Power < 0 and falling (making lower high) AND price < 12h EMA50 AND 1d ADX > 25. 
# Exits when Elder Ray power crosses zero OR ADX < 20 (regime shift to ranging). 
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness in trending markets (both bull and bear regimes) by capturing institutional participation via Elder Ray while avoiding whipsaws in ranging markets via ADX filter.

name = "6h_ElderRay_TrendFilter_ADXRegime_v1"
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
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d ADX for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need enough for ADX calculation
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
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # first value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(tr) < period:
        return np.zeros(n)
        
    atr = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    # Avoid division by zero
    divisor = np.where(atr == 0, 1, atr)
    plus_di = 100 * plus_dm_smooth / divisor
    minus_di = 100 * minus_dm_smooth / divisor
    
    dx = np.where((plus_di + minus_di) == 0, 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = wilder_smooth(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Elder Ray (Bull Power/Bear Power) using EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND making higher low (bullish momentum) AND price > 12h EMA50 AND ADX > 25 (trending)
            if (bull_power[i] > 0 and 
                i > 20 and bull_power[i] > bull_power[i-1] and  # making higher low
                close[i] > ema_50_12h_aligned[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND making lower high (bearish momentum) AND price < 12h EMA50 AND ADX > 25 (trending)
            elif (bear_power[i] < 0 and 
                  i > 20 and bear_power[i] < bear_power[i-1] and  # making lower high
                  close[i] < ema_50_12h_aligned[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power crosses below zero OR ADX < 20 (regime shift to ranging)
            if (bull_power[i] <= 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power crosses above zero OR ADX < 20 (regime shift to ranging)
            if (bear_power[i] >= 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals