#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly Bollinger Band squeeze breakout + volume confirmation.
# In both bull and bear markets, low volatility periods (BB width < 20th percentile) precede explosive moves.
# Breakout direction confirmed by 1-week EMA50 trend filter. Uses weekly data for structure, 6h for entry.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.
# Weekly Bollinger squeeze acts as a volatility filter that works across regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Weekly data for Bollinger Bands and EMA50 trend ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Bollinger Bands (20, 2.0)
    bb_mid_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    bb_std_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    bb_upper_1w = bb_mid_1w + 2.0 * bb_std_1w
    bb_lower_1w = bb_mid_1w - 2.0 * bb_std_1w
    bb_width_1w = bb_upper_1w - bb_lower_1w  # Absolute width
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to 6h timeframe
    bb_width_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_width_1w)
    bb_upper_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_upper_1w)
    bb_lower_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_lower_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly Bollinger width percentile (20-period lookback) for squeeze detection
    bb_width_percentile = pd.Series(bb_width_1w_aligned).rolling(
        window=20, min_periods=20
    ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False).values
    
    # === 1d volume for confirmation (more responsive than weekly) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for weekly indicators
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bb_width_percentile[i]) or 
            np.isnan(bb_upper_1w_aligned[i]) or
            np.isnan(bb_lower_1w_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # Volume confirmation: current day's volume > 1.5x 20-day average
        # Get current day's volume aligned to 6h
        df_1d_current = get_htf_data(prices, '1d')
        vol_1d_current = df_1d_current['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d_current, vol_1d_current)
        vol_confirm = vol_1d_aligned[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Squeeze condition: BB width < 20th percentile (low volatility)
        squeeze = bb_width_percentile[i] < 20.0
        
        # Trend filter: price above/below weekly EMA50
        above_ema50 = price > ema50_1w_aligned[i]
        below_ema50 = price < ema50_1w_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: squeeze breakout above upper BB + volume + above EMA50
            if squeeze and price > bb_upper_1w_aligned[i] and vol_confirm and above_ema50:
                signals[i] = 0.25
                position = 1
                continue
            # Short: squeeze breakout below lower BB + volume + below EMA50
            elif squeeze and price < bb_lower_1w_aligned[i] and vol_confirm and below_ema50:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite squeeze breakout
        elif position == 1:
            # Exit long if squeeze breakout below lower BB
            if squeeze and price < bb_lower_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if squeeze breakout above upper BB
            if squeeze and price > bb_upper_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyBB_SqueezeBreakout_Volume1.5x_EMA50Filter"
timeframe = "6h"
leverage = 1.0