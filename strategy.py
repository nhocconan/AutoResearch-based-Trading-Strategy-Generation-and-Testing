#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long when Bull Power > 0 AND ADX(14) > 25 AND +DI > -DI (strong uptrend)
# - Short when Bear Power < 0 AND ADX(14) > 25 AND -DI > +DI (strong downtrend)
# - Exit when ADX < 20 (trend weakening) or Elder Power reverses
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets by trading with the strong trend (ADX > 25)

name = "6h_1d_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute 6h EMA(13) for Elder Ray
    def ema(arr, span):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < span:
            return result
        multiplier = 2 / (span + 1)
        result[span-1] = np.mean(arr[:span])  # SMA seed
        for i in range(span, len(arr)):
            result[i] = (arr[i] * multiplier) + (result[i-1] * (1 - multiplier))
        return result
    
    ema_13 = ema(close, 13)
    
    # Pre-compute 6h Elder Ray
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Pre-compute 1d ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros_like(high_1d)
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # +DM and -DM
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # positive values
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Wilder's smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    # +DI and -DI
    plus_di = np.where(atr_1d != 0, 100 * plus_dm_smooth / atr_1d, 0)
    minus_di = np.where(atr_1d != 0, 100 * minus_dm_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = wilders_smooth(dx, 14)
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power positive AND strong uptrend (ADX>25 and +DI > -DI)
            if (bull_power[i] > 0 and 
                adx_aligned[i] > 25 and 
                plus_di_aligned[i] > minus_di_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power negative AND strong downtrend (ADX>25 and -DI > +DI)
            elif (bear_power[i] < 0 and 
                  adx_aligned[i] > 25 and 
                  minus_di_aligned[i] > plus_di_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: trend weakening (ADX < 20) or Elder Power reverses
            exit_long = (position == 1 and 
                        (adx_aligned[i] < 20 or bull_power[i] <= 0))
            exit_short = (position == -1 and 
                         (adx_aligned[i] < 20 or bear_power[i] >= 0))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals