#!/usr/bin/env python3
# 6H_ADX_DMI_Cross_1dTrend_Filter
# Hypothesis: Uses ADX(14) and DMI crossover on 6h chart filtered by 1-day trend (close > EMA50).
# Enters long when +DI crosses above -DI with ADX > 25 in uptrend (close > EMA50).
# Enters short when -DI crosses above +DI with ADX > 25 in downtrend (close < EMA50).
# Exits when DMI reverses or ADX falls below 20.
# Uses 1-day EMA50 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 12-37 trades per year on 6h timeframe with position size 0.25.

name = "6H_ADX_DMI_Cross_1dTrend_Filter"
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
    
    # Get 1d data for trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ADX and DMI on 6h chart
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth(values, period):
        smoothed = np.zeros_like(values)
        if len(values) < period:
            return smoothed
        smoothed[period-1] = np.sum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    period = 14
    atr = smooth(tr, period)
    plus_di_smoothed = smooth(plus_dm, period)
    minus_di_smoothed = smooth(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, plus_di_smoothed / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_di_smoothed / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = smooth(dx, period)
    
    # Align 1d EMA50 to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2*period)  # Warmup for smoothing
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # DMI crossover signals
        plus_cross_above = (plus_di[i] > minus_di[i]) and (plus_di[i-1] <= minus_di[i-1])
        minus_cross_above = (minus_di[i] > plus_di[i]) and (minus_di[i-1] <= plus_di[i-1])
        
        if position == 0:
            # Long entry: +DI crosses above -DI with ADX > 25 in uptrend
            if (plus_cross_above and 
                adx[i] > 25 and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: -DI crosses above +DI with ADX > 25 in downtrend
            elif (minus_cross_above and 
                  adx[i] > 25 and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: -DI crosses above +DI or ADX falls below 20
            if (minus_cross_above or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: +DI crosses above -DI or ADX falls below 20
            if (plus_cross_above or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals