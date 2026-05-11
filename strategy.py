#!/usr/bin/env python3
# 6h_WeeklyPivot_Rejection_Volume
# Hypothesis: Uses weekly pivot points as key support/resistance levels. 
# Enters on price rejection (long wicks) at S1/R1 levels with volume confirmation.
# Uses daily trend filter to avoid counter-trend trades.
# Works in ranging markets (pivot bounces) and trending markets (breakouts).
# Weekly pivots provide structure that adapts to volatility.

name = "6h_WeeklyPivot_Rejection_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points (calculated from prior week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly pivot points (using prior week's OHLC) ---
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivots using prior week's data (no look-ahead)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # --- Daily trend filter (HH/HL for uptrend, LH/LL for downtrend) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    hh = high_1d > np.roll(high_1d, 1)
    hl = low_1d > np.roll(low_1d, 1)
    lh = high_1d < np.roll(high_1d, 1)
    ll = low_1d < np.roll(low_1d, 1)
    uptrend = hh & hl
    downtrend = lh & ll
    uptrend[0] = False
    downtrend[0] = False
    
    # --- Volume confirmation (volume > 1.5x 24-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    # --- Price rejection detection (long wicks) ---
    # Bullish rejection: long lower wick (close near high, low far below)
    # Bearish rejection: long upper wick (close near low, high far above)
    body_size = np.abs(close - open_prices) if 'open_prices' in locals() else np.abs(close - np.roll(close, 1))
    lower_wick = np.minimum(close, open_prices) - low if 'open_prices' in locals() else np.minimum(close, np.roll(close, 1)) - low
    upper_wick = high - np.maximum(close, open_prices) if 'open_prices' in locals() else high - np.maximum(close, np.roll(close, 1))
    
    # Handle first bar
    if 'open_prices' not in locals():
        open_prices = np.roll(close, 1)
        open_prices[0] = close[0]
        body_size = np.abs(close - open_prices)
        lower_wick = np.minimum(close, open_prices) - low
        upper_wick = high - np.maximum(close, open_prices)
    
    # Normalize wicks by ATR-like measure to avoid scale issues
    true_range = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    true_range[0] = high[0] - low[0]
    atr_like = np.full(n, np.nan)
    for i in range(10, n):
        atr_like[i] = np.mean(true_range[i-10:i])
    
    # Significant wick rejection (wick > 2x body AND significant compared to volatility)
    bullish_rejection = (lower_wick > 2 * body_size) & (lower_wick > atr_like * 0.5)
    bearish_rejection = (upper_wick > 2 * body_size) & (upper_wick > atr_like * 0.5)
    
    # Align all timeframe data to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for weekly pivot calculation and volume MA
    start_idx = max(24, 10)  # volume MA(24) and ATR-like(10)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_1w_aligned[i]) or
            np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i]) or
            np.isnan(atr_like[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current pivot levels
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long setup: bullish rejection at support levels in uptrend or at S1/S2 in any trend
            if bullish_rejection[i] and vol_spike:
                near_s1 = abs(low[i] - s1) < (pivot - s1) * 0.3  # Within 30% of S1
                near_s2 = abs(low[i] - s2) < (pivot - s2) * 0.3  # Within 30% of S2
                in_uptrend = uptrend_aligned[i]
                
                if (near_s1 or near_s2) and in_uptrend:
                    # In uptrend, buy pullbacks to support
                    signals[i] = 0.25
                    position = 1
                elif near_s1 or near_s2:
                    # In ranging/downtrend, buy at support for mean reversion
                    signals[i] = 0.20
                    position = 1
            
            # Short setup: bearish rejection at resistance levels in downtrend or at R1/R2 in any trend
            elif bearish_rejection[i] and vol_spike:
                near_r1 = abs(high[i] - r1) < (r1 - pivot) * 0.3  # Within 30% of R1
                near_r2 = abs(high[i] - r2) < (r2 - pivot) * 0.3  # Within 30% of R2
                in_downtrend = downtrend_aligned[i]
                
                if (near_r1 or near_r2) and in_downtrend:
                    # In downtrend, sell rallies to resistance
                    signals[i] = -0.25
                    position = -1
                elif near_r1 or near_r2:
                    # In ranging/uptrend, sell at resistance for mean reversion
                    signals[i] = -0.20
                    position = -1
        
        else:
            if position == 1:
                # Exit long: price reaches pivot, shows bearish rejection, or trend breaks
                if (close[i] >= pivot or 
                    bearish_rejection[i] and vol_spike or 
                    not uptrend_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches pivot, shows bullish rejection, or trend breaks
                if (close[i] <= pivot or 
                    bullish_rejection[i] and vol_spike or 
                    not downtrend_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals