#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Breakout with Volume and ATR Filter
# Uses weekly pivot points (calculated from weekly high/low/close) to identify key support/resistance
# Enters long when price breaks above weekly R1 with volume > 1.5x 20-week average and ATR(14) < 0.02*price
# Enters short when price breaks below weekly S1 with volume > 1.5x 20-week average and ATR(14) < 0.02*price
# Exits when price returns to weekly pivot point (mean reversion to equilibrium)
# Weekly pivots provide institutional reference points; volume filter ensures conviction; ATR filter avoids choppy markets
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

name = "6h_WeeklyPivot_Breakout_VolumeATR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for pivot points and filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Calculate 20-week average volume
    volume_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on weekly data for volatility filter
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # First TR undefined
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all weekly data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Get 6s data for entry signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_20_val = vol_ma_20_aligned[i]
        atr_14_val = atr_14_aligned[i]
        
        # Skip if any value is NaN or invalid
        if (np.isnan(pivot_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(vol_ma_20_val) or np.isnan(atr_14_val) or 
            np.isnan(close_val) or np.isnan(volume_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extremely choppy markets (ATR < 2% of price)
        vol_filter = atr_14_val < 0.02 * close_val
        
        # Volume filter: current volume > 1.5x 20-week average (scaled to 6h)
        # Approximate: 1 week ≈ 28 * 6h bars (7 days * 4 bars/day)
        vol_threshold = vol_ma_20_val / 28.0 * 1.5
        volume_filter = volume_val > vol_threshold
        
        if position == 0:
            # Long entry: price breaks above weekly R1 with volume and volatility filters
            if close_val > r1_val and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1 with volume and volatility filters
            elif close_val < s1_val and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly pivot (mean reversion)
            if close_val <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly pivot (mean reversion)
            if close_val >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals