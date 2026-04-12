#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculations and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 50-period EMA on daily for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate daily pivots (standard floor pivot)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Calculate 4-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr4 = np.full(n, np.nan)
    for i in range(3, n):
        atr4[i] = np.nanmean(tr[i-3:i+1])
    
    # Calculate 4-period ATR EMA for volatility regime
    atr_series = pd.Series(atr4)
    atr_ema4 = atr_series.ewm(span=4, adjust=False, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(atr4[i]) or np.isnan(atr_ema4[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR4 > 0.8x 4-period ATR EMA (avoid low volatility)
        vol_filter = atr4[i] > atr_ema4[i] * 0.8
        
        # Trend filter: price above/below daily 50 EMA
        price_above_ema50 = close[i] > ema50_1d_aligned[i]
        price_below_ema50 = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions: bounce from S1/S2 with trend alignment
        long_bounce_s1 = (low[i] <= s1_aligned[i] * 1.005) and (close[i] > s1_aligned[i])
        long_bounce_s2 = (low[i] <= s2_aligned[i] * 1.005) and (close[i] > s2_aligned[i])
        long_entry = (long_bounce_s1 or long_bounce_s2) and price_above_ema50 and vol_filter
        
        # Entry conditions: rejection at R1/R2 with trend alignment
        short_reject_r1 = (high[i] >= r1_aligned[i] * 0.995) and (close[i] < r1_aligned[i])
        short_reject_r2 = (high[i] >= r2_aligned[i] * 0.995) and (close[i] < r2_aligned[i])
        short_entry = (short_reject_r1 or short_reject_r2) and price_below_ema50 and vol_filter
        
        # Exit conditions: opposite signal or volatility drop
        long_exit = (close[i] < ema50_1d_aligned[i]) or (atr4[i] < atr_ema4[i] * 0.6)
        short_exit = (close[i] > ema50_1d_aligned[i]) or (atr4[i] < atr_ema4[i] * 0.6)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_pivot_ema50_vol_filter_v1"
timeframe = "4h"
leverage = 1.0