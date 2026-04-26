#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_ChopFilter_v1
Hypothesis: Trade Camarilla R3/S3 breakouts on 12h timeframe with 1-week EMA trend filter and choppiness regime filter.
Uses ATR trailing stop (1.5x) for risk management. Designed for low-frequency, high-conviction trades
that work in both bull and bear markets by requiring: 1) strong HTF trend alignment (1w EMA50),
2) breakout of significant pivot levels (R3/S3), 3) non-choppy market conditions (Chop < 61.8).
Target: 20-50 trades over 4 years to minimize fee drag while capturing major moves.
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
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align HTF indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Choppiness Index on 1d timeframe (regime filter)
    hl_range = np.maximum(df_1d['high'].values, df_1d['low'].shift(1).values) - np.minimum(df_1d['low'].values, df_1d['high'].shift(1).values)
    atr_1d = pd.Series(hl_range).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # ATR for stop (14-period on 12h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1w EMA (50), 1d ATR (14), chop (14), 12h ATR (14)
    start_idx = max(50, 14, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        chop_val = chop_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: break above R3, uptrend (close > EMA50_1w), non-choppy (Chop < 61.8)
            long_signal = (high_val > camarilla_r3_val) and \
                          (close_val > ema_50_1w_val) and \
                          (chop_val < 61.8)
            # Short: break below S3, downtrend (close < EMA50_1w), non-choppy (Chop < 61.8)
            short_signal = (low_val < camarilla_s3_val) and \
                           (close_val < ema_50_1w_val) and \
                           (chop_val < 61.8)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 1.5 * atr_14_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 1.5 * atr_14_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 1.5 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < EMA50_1w) after minimum holding period
            if bars_since_entry >= 4 and ((low_val < long_stop) or (close_val < ema_50_1w_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 1.5 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50_1w) after minimum holding period
            if bars_since_entry >= 4 and ((high_val > short_stop) or (close_val > ema_50_1w_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0