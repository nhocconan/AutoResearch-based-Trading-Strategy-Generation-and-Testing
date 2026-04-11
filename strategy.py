#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_trend_v1
# Strategy: Daily Camarilla pivot with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Uses weekly trend via EMA50 on weekly closes to filter daily reversals at S3/R3 and breakouts at S4/R4. Volume confirms entries. Works in bull/bear by aligning with higher timeframe trend while capturing reversals or breakouts. Targets ~50 trades over 4 years (~12/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_trend_v1"
timeframe = "1d"
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
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly OHLC for EMA50 trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe (wait for weekly close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily OHLC for Camarilla pivots (using previous day's data)
    # We'll compute daily pivots from daily data, then align as needed
    # But since we're on daily timeframe, we can compute directly
    # However, we need previous day's levels, so we shift by 1
    
    # Calculate daily pivots using previous day's OHLC
    # We'll compute arrays for H, L, C of previous day
    # For index i, we use day i-1's OHLC
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    # Set first value to NaN since no previous day
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot = (high_prev + low_prev + close_prev) / 3.0
    # Range = H - L
    range_val = high_prev - low_prev
    # R4 = C + (H-L) * 1.1/2
    r4 = close_prev + range_val * 1.1 / 2.0
    # R3 = C + (H-L) * 1.1/4
    r3 = close_prev + range_val * 1.1 / 4.0
    # S3 = C - (H-L) * 1.1/4
    s3 = close_prev - range_val * 1.1 / 4.0
    # S4 = C - (H-L) * 1.1/2
    s4 = close_prev - range_val * 1.1 / 2.0
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3[i]) or np.isnan(r4[i]) or 
            np.isnan(s3[i]) or np.isnan(s4[i]) or np.isnan(pivot[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA50
        uptrend_1w = price_close > ema_50_1w_aligned[i]
        downtrend_1w = price_close < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: Reversal at S3 with volume in uptrend OR break above R4 with volume in uptrend
        long_reversal = price_close > s3[i] and price_close < s3[i-1]  # Crossing above S3 from below
        long_breakout = price_close > r4[i]  # Break above R4
        long_signal = (long_reversal and vol_confirmed and uptrend_1w) or \
                      (long_breakout and vol_confirmed and uptrend_1w)
        
        # Short: Reversal at R3 with volume in downtrend OR break below S4 with volume in downtrend
        short_reversal = price_close < r3[i] and price_close > r3[i-1]  # Crossing below R3 from above
        short_breakout = price_close < s4[i]  # Break below S4
        short_signal = (short_reversal and vol_confirmed and downtrend_1w) or \
                       (short_breakout and vol_confirmed and downtrend_1w)
        
        # Exit when price returns to the daily pivot level or opposite Camarilla level
        exit_long = position == 1 and (price_close < pivot[i] or price_close < s3[i])
        exit_short = position == -1 and (price_close > pivot[i] or price_close > r3[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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