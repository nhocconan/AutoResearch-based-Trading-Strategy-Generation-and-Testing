#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_regime_v1
# Hypothesis: 4-hour Camarilla breakout with volume confirmation and chop regime filter
# Uses daily Camarilla levels from prior day, volume > 1.5x 20-bar average, and chop > 61.8 (range) for mean-reversion logic
# Works in bull/bear by fading extremes in ranging markets and avoiding false breakouts in trends
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag

name = "4h_1d_camarilla_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first bar
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else high_1d[0]
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else low_1d[0]
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else close_1d[0]
    
    # Camarilla levels (based on previous day)
    range_ = prev_high - prev_low
    # Resistance levels
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    # Support levels
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # Chop regime: chop > 61.8 = range (mean revert)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    denom = max_hh - min_ll
    chop = np.where(denom != 0, 100 * np.log10(atr_14.rolling(14, min_periods=1).sum() / denom) / np.log10(14), 61.8)
    chop = np.where(np.isnan(chop), 61.8, chop)
    
    # Align Camarilla levels and chop to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (chop > 61.8)
        if chop_aligned[i] <= 61.8:
            # In trending markets, stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Long entry: close breaks above S4 (mean reversion to S3)
        if (close[i] < s4_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks above R4 (mean reversion to R3)
        elif (close[i] > r4_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: mean reversion to opposite S3/R3
        elif position == 1 and close[i] > s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < r3_aligned[i]:
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