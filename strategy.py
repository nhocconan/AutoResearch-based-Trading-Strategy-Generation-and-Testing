#!/usr/bin/env python3
"""
12h_1W_Camarilla_Pivot_Breakout_1DTrend_Volume_v1
Hypothesis: Uses weekly pivot points as structural support/resistance levels, with breakout
confirmation from 12h price action, 1d trend filter, and volume confirmation. Designed for
low frequency (15-25 trades/year) to work in both bull (breakouts above weekly pivots) and
bear (breakdowns below weekly pivots) markets. Weekly pivots provide stronger levels than
daily pivots, reducing false breakouts. Volume confirmation ensures breakouts have
institutional participation. 1d trend filter avoids counter-trend trades.
"""

name = "12h_1W_Camarilla_Pivot_Breakout_1DTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    r2 = close + range_val * 1.1 / 6
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return r4, r3, r2, r1, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1w Camarilla pivots (updated weekly) ---
    # Calculate pivots for each weekly bar
    r4_1w = np.full(len(df_1w), np.nan)
    r3_1w = np.full(len(df_1w), np.nan)
    r2_1w = np.full(len(df_1w), np.nan)
    r1_1w = np.full(len(df_1w), np.nan)
    s1_1w = np.full(len(df_1w), np.nan)
    s2_1w = np.full(len(df_1w), np.nan)
    s3_1w = np.full(len(df_1w), np.nan)
    s4_1w = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        r4, r3, r2, r1, s1, s2, s3, s4 = calculate_camarilla_pivots(
            df_1w['high'].iloc[i], df_1w['low'].iloc[i], df_1w['close'].iloc[i]
        )
        r4_1w[i] = r4
        r3_1w[i] = r3
        r2_1w[i] = r2
        r1_1w[i] = r1
        s1_1w[i] = s1
        s2_1w[i] = s2
        s3_1w[i] = s3
        s4_1w[i] = s4
    
    # Align weekly pivots to 12h timeframe (wait for weekly bar to close)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # --- 1d EMA34 for trend filter ---
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume confirmation ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(r2_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma.iloc[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above R3 with volume, in uptrend
        long_signal = (high[i] > r3_1w_aligned[i]) and vol_spike[i] and (close[i] > ema_34_1d_aligned[i])
        
        # Short signal: price breaks below S3 with volume, in downtrend
        short_signal = (low[i] < s3_1w_aligned[i]) and vol_spike[i] and (close[i] < ema_34_1d_aligned[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: opposite signal or reversion to pivot
            if position == 1:
                # Exit long if price breaks below R1 or reverses to S1
                exit_signal = (low[i] < r1_1w_aligned[i]) or (close[i] < s1_1w_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if price breaks above S1 or reverses to R1
                exit_signal = (high[i] > s1_1w_aligned[i]) or (close[i] > r1_1w_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals