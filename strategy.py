# 6h_1d_1w_camarilla_breakout_v1
# Strategy: 6h Camarilla pivot breakout with weekly trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Uses 1d Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) as entry signals,
# filtered by 1w EMA50 trend alignment. Works in both bull and bear markets by following higher timeframe trend.
# Targets 10-30 trades per year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # Range = H - L
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3_1d = pivot_1d + (range_1d * 1.1 / 2)
    s3_1d = pivot_1d - (range_1d * 1.1 / 2)
    r4_1d = pivot_1d + (range_1d * 1.1)
    s4_1d = pivot_1d - (range_1d * 1.1)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align Camarilla levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1w EMA50
        uptrend_1w = price_close > ema_50_1w_aligned[i]
        downtrend_1w = price_close < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Fade at R3/S3 (mean reversion)
        fade_long = price_close <= s3_1d_aligned[i] and vol_confirmed and uptrend_1w
        fade_short = price_close >= r3_1d_aligned[i] and vol_confirmed and downtrend_1w
        
        # Breakout continuation at R4/S4 (trend following)
        breakout_long = price_close >= r4_1d_aligned[i] and vol_confirmed and uptrend_1w
        breakout_short = price_close <= s4_1d_aligned[i] and vol_confirmed and downtrend_1w
        
        # Exit conditions: return to pivot or opposite extreme
        exit_long = position == 1 and (price_close <= pivot_1d_aligned[i] or price_close >= r4_1d_aligned[i])
        exit_short = position == -1 and (price_close >= pivot_1d_aligned[i] or price_close <= s4_1d_aligned[i])
        
        # Trading logic
        if (fade_long or breakout_long) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (fade_short or breakout_short) and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals