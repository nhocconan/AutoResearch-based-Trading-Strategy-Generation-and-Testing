#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly pivot points (S1/R1, S2/R2, S3/R3, S4/R4) on 1w timeframe
# combined with 1d trend filter (EMA50) and volume confirmation on 6h timeframe.
# Enters on breakouts beyond S3/R3 with trend alignment, exits at S4/R4 or reversal.
# Designed for low frequency (target 15-30 trades/year) to avoid fee drag in both bull and bear markets.
# Weekly pivots provide strong institutional levels; 1d EMA50 filters trend; volume confirms conviction.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    pivot_1w = np.full(len(high_1w), np.nan)
    r1_1w = np.full(len(high_1w), np.nan)
    s1_1w = np.full(len(high_1w), np.nan)
    r2_1w = np.full(len(high_1w), np.nan)
    s2_1w = np.full(len(high_1w), np.nan)
    r3_1w = np.full(len(high_1w), np.nan)
    s3_1w = np.full(len(high_1w), np.nan)
    r4_1w = np.full(len(high_1w), np.nan)
    s4_1w = np.full(len(high_1w), np.nan)
    
    for i in range(1, len(high_1w)):
        # Use previous week's data to calculate current week's pivots
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        pp = (ph + pl + pc) / 3.0
        pivot_1w[i] = pp
        
        r1_1w[i] = 2 * pp - pl
        s1_1w[i] = 2 * pp - ph
        
        r2_1w[i] = pp + (ph - pl)
        s2_1w[i] = pp - (ph - pl)
        
        r3_1w[i] = ph + 2 * (pp - pl)
        s3_1w[i] = pl - 2 * (ph - pp)
        
        r4_1w[i] = pp + 3 * (ph - pl)
        s4_1w[i] = pp - 3 * (ph - pl)
    
    # Align weekly pivot levels to 6h timeframe (no extra delay needed as pivots are known at week start)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on daily close
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])  # SMA for first value
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_50_1d[i-1] * (49 / (50 + 1)))
    
    # Align daily EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume moving average (50-period) on 6h
    vol_ma = np.full(n, np.nan)
    for i in range(50, n):
        vol_ma[i] = np.mean(volume[i-50:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # need weekly pivot (from i=1) and daily EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * 50-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: price above/below daily EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: break above R3 with volume and uptrend
            if (close[i] > r3_1w_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: break below S3 with volume and downtrend
            elif (close[i] < s3_1w_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: reverse below S3 or reach R4
            if close[i] < s3_1w_aligned[i] or close[i] > r4_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reverse above R3 or reach S4
            if close[i] > r3_1w_aligned[i] or close[i] < s4_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_S3R3_Breakout_EMA50_Volume"
timeframe = "6h"
leverage = 1.0