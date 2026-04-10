#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter with 1d/1w HTF confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - ADX > 25 indicates trending market (use 1d ADX for regime)
# - Long when Bull Power > 0 AND ADX > 25 AND price > EMA20 (1d)
# - Short when Bear Power < 0 AND ADX > 25 AND price < EMA20 (1d)
# - Exit when Elder Ray power reverses sign OR ADX < 20
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray captures bull/bear strength relative to EMA
# - ADX filter ensures we trade only in trending regimes
# - 1d EMA20 provides higher timeframe trend filter

name = "6h_1d_elder_ray_adx_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Pre-compute 6h EMA20 for trend filter
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Pre-compute 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/14)
    def wilders_smooth(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.mean(arr[1:period+1])  # First value
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = np.zeros_like(atr_1d)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx_1d = wilders_smooth(dx, 14)
    
    # Pre-compute 1d EMA20 for trend filter
    close_1d_s = pd.Series(close_1d)
    ema20_1d = close_1d_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Pre-compute 1w trend filter (price above/below 1w EMA20)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema20_1w = close_1w_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(ema20[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND ADX > 25 AND price > EMA20 (1d) AND price > EMA20 (1w)
            if (bull_power[i] > 0 and 
                adx_1d_aligned[i] > 25 and 
                close[i] > ema20_1d_aligned[i] and
                close[i] > ema20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power < 0 AND ADX > 25 AND price < EMA20 (1d) AND price < EMA20 (1w)
            elif (bear_power[i] < 0 and 
                  adx_1d_aligned[i] > 25 and 
                  close[i] < ema20_1d_aligned[i] and
                  close[i] < ema20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Elder Ray power reverses sign OR ADX < 20 OR price crosses EMA20 (1d)
            exit_long = (position == 1 and (bull_power[i] <= 0 or adx_1d_aligned[i] < 20 or close[i] < ema20_1d_aligned[i]))
            exit_short = (position == -1 and (bear_power[i] >= 0 or adx_1d_aligned[i] < 20 or close[i] > ema20_1d_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals