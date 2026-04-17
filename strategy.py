#!/usr/bin/env python3
"""
6h_ADX_DMI_Trend_with_1d_Pullback
Hypothesis: On 6h, enter long when ADX > 25 and +DI > -DI (strong uptrend) and price pulls back to EMA20; short when ADX > 25 and -DI > +DI and price pulls back to EMA20. Uses 1d EMA50 filter to align with higher timeframe trend and avoid counter-trend trades. Designed for 20-40 trades/year to minimize fee drag and work in both bull/bear regimes via trend alignment and pullback entries.
"""

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
    
    # === 6h ADX/DMI calculation ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    tr_sum = wilder_smooth(tr, period)
    plus_di_sum = wilder_smooth(plus_dm, period)
    minus_di_sum = wilder_smooth(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_sum != 0, 100 * plus_di_sum / tr_sum, 0)
    minus_di = np.where(tr_sum != 0, 100 * minus_di_sum / tr_sum, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smooth(dx, period)
    
    # === 6h EMA20 for pullback entries ===
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup covers ADX/DMI calculation (2*period) and EMA20
    warmup = 2 * period + 20
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(adx[i]) or 
            np.isnan(plus_di[i]) or 
            np.isnan(minus_di[i]) or 
            np.isnan(ema20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: ADX > 25, +DI > -DI (uptrend), price at or near EMA20 (pullback)
            if (adx[i] > 25 and 
                plus_di[i] > minus_di[i] and 
                close[i] <= ema20[i] * 1.005 and  # within 0.5% above EMA20
                close[i] > ema50_1d_aligned[i]):   # above 1d EMA50 (long-term uptrend)
                signals[i] = 0.25
                position = 1
                continue
            # Short: ADX > 25, -DI > +DI (downtrend), price at or near EMA20 (pullback)
            elif (adx[i] > 25 and 
                  minus_di[i] > plus_di[i] and 
                  close[i] >= ema20[i] * 0.995 and  # within 0.5% below EMA20
                  close[i] < ema50_1d_aligned[i]):   # below 1d EMA50 (long-term downtrend)
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: trend weakening or opposite signal
        elif position == 1:
            # Exit long if ADX weakens or trend reverses
            if adx[i] < 20 or minus_di[i] > plus_di[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if ADX weakens or trend reverses
            if adx[i] < 20 or plus_di[i] > minus_di[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_DMI_Trend_with_1d_Pullback"
timeframe = "6h"
leverage = 1.0