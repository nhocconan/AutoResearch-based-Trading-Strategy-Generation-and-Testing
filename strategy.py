#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R3/S3 breakout with 1d trend filter and volume confirmation.
# Uses 1d EMA(50) for trend bias and Camarilla levels from prior 1d for breakout entries.
# Includes volume filter (current volume > 1.8x 20-bar SMA) to avoid false breakouts.
# Designed for low trade frequency (20-40/year) to minimize fee drag.
# Works in bull/bear: 1d EMA avoids counter-trend trades, Camarilla provides structure, volume confirms momentum.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Camarilla Pivot Levels from prior 1d ===
    # Calculate Camarilla levels using prior 1d OHLC
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use prior 1d to avoid look-ahead
    close_1d = pd.Series(df_1d['close'].values)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    
    # Prior 1d values (shifted by 1 to avoid look-ahead)
    prior_close = close_1d.shift(1).values
    prior_high = high_1d.shift(1).values
    prior_low = low_1d.shift(1).values
    
    # Calculate Camarilla levels for prior 1d
    range_1d = prior_high - prior_low
    r3 = prior_close + 1.1 * range_1d
    s3 = prior_close - 1.1 * range_1d
    r4 = prior_close + 1.5 * range_1d
    s4 = prior_close - 1.5 * range_1d
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current 4h volume > 1.8x 20-period 4h volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above R3 (bullish breakout)
        # 2. 1d price above EMA50 (bullish long-term trend bias)
        # 3. Volume confirmation
        if (close[i] > r3_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below S3 (bearish breakout)
        # 2. 1d price below EMA50 (bearish long-term trend bias)
        # 3. Volume confirmation
        elif (close[i] < s3_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_EMA50_VolFilter_v1"
timeframe = "4h"
leverage = 1.0