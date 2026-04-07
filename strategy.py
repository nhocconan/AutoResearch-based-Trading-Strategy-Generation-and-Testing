#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla pivot with daily volume and weekly trend filter
# Hypothesis: Camarilla levels provide high-probability reversal points in ranging markets.
# Volume confirms institutional interest, weekly trend filter avoids counter-trend trades.
# Works in bull via mean reversion at support/resistance, in bear via trend-aligned entries.
# Target: 15-30 trades/year to minimize fee drag on 12h timeframe.
name = "12h_camarilla_pivot_1d_volume_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    camarilla_h4 = typical_price + (1.1 * range_val / 2)
    camarilla_l4 = typical_price - (1.1 * range_val / 2)
    camarilla_h3 = typical_price + (1.1 * range_val / 4)
    camarilla_l3 = typical_price - (1.1 * range_val / 4)
    camarilla_h2 = typical_price + (1.1 * range_val / 6)
    camarilla_l2 = typical_price - (1.1 * range_val / 6)
    camarilla_h1 = typical_price + (1.1 * range_val / 12)
    camarilla_l1 = typical_price - (1.1 * range_val / 12)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF data to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4.values)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4.values)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2.values)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2.values)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1.values)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1.values)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Weekly trend filter: price above/below EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price reaches L3 level or trend changes
            if close[i] <= camarilla_l3_aligned[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches H3 level or trend changes
            if close[i] >= camarilla_h3_aligned[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price touches S1/L1 level with volume confirmation in uptrend
            if (close[i] <= camarilla_l1_aligned[i] and vol_confirm and uptrend):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches R1/H1 level with volume confirmation in downtrend
            elif (close[i] >= camarilla_h1_aligned[i] and vol_confirm and downtrend):
                position = -1
                signals[i] = -0.25
    
    return signals