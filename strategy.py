#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot (R1/S1) breakout with volume confirmation and 4h ADX regime filter.
# In bull/bear markets, price breaking above R1 or below S1 on 1d chart indicates strong momentum.
# Volume confirms institutional participation. ADX > 25 ensures trending regime to avoid false breakouts in chop.
# Designed for low trade frequency (20-40/year) to minimize fee drag while capturing strong directional moves.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h Indicators: ADX(14) for trend regime ===
    # +DI, -DI, DX calculation
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low[::-1])[::-1]  # reversed for proper alignment
    down_move = np.append(down_move[1:], down_move[-1])  # fix length
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first period
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    period = 14
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
    
    plus_di = 100 * plus_dm_sum / (tr_sum + 1e-10)
    minus_di = 100 * minus_dm_sum / (tr_sum + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d R1 (resistance)
        # 2. Volume confirmation
        # 3. ADX > 25 (trending regime)
        if (close[i] > r1_aligned[i]) and vol_confirm and (adx_aligned[i] > 25):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d S1 (support)
        # 2. Volume confirmation
        # 3. ADX > 25 (trending regime)
        elif (close[i] < s1_aligned[i]) and vol_confirm and (adx_aligned[i] > 25):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R1S1_Volume_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0