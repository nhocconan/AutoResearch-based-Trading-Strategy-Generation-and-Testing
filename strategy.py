#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power rising (improving) AND 12h ADX > 25 (trending market)
# - Short when Bear Power > 0 AND Bull Power falling (worsening) AND 12h ADX > 25
# - Exit when power signals reverse OR ADX drops below 20 (range market)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear strength relative to trend (EMA13)
# - ADX filter ensures we only trade in trending conditions where Elder Ray works best
# - Works in both bull (strong Bull Power) and bear (strong Bear Power) markets

name = "6h_12h_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA
    bear_power = ema13 - low   # Bear Power = EMA - Low
    
    # Pre-compute 6h Elder Ray momentum (change in power)
    bull_power_momentum = np.diff(bull_power, prepend=bull_power[0])
    bear_power_momentum = np.diff(bear_power, prepend=bear_power[0])
    
    # Pre-compute 12h ADX for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_12h = np.zeros_like(tr)
    atr_12h[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    # Directional Movement
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = -np.diff(low_12h, prepend=low_12h[0])  # negative of low change
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and ATR
    def wilders_smoothing(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.mean(arr[1:period])  # First value
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    period = 14
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    atr_smooth = wilders_smoothing(tr, period)
    
    # Avoid division by zero
    atr_smooth[atr_smooth == 0] = 1e-10
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = np.zeros_like(dx)
    adx[period-1] = np.mean(dx[1:period])  # First ADX value
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_momentum[i]) or np.isnan(bear_power_momentum[i]) or
            np.isnan(adx_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power positive AND rising AND strong trend (ADX > 25)
            if (bull_power[i] > 0 and 
                bull_power_momentum[i] > 0 and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power positive AND rising AND strong trend (ADX > 25)
            elif (bear_power[i] > 0 and 
                  bear_power_momentum[i] > 0 and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: power signals reverse OR ADX drops below 20 (range market)
            exit_long = (position == 1 and 
                        (bull_power[i] <= 0 or bull_power_momentum[i] <= 0 or adx_aligned[i] < 20))
            exit_short = (position == -1 and 
                          (bear_power[i] <= 0 or bear_power_momentum[i] <= 0 or adx_aligned[i] < 20))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals