#!/usr/bin/env python3
"""
1d_Williams_Alligator_1wTrend_HTFRegime_v1
Hypothesis: Use Williams Alligator on 1d for trend direction and entry timing, with 1w HTF regime filter.
In bull 1w regime (price > 1w EMA50): go long when Alligator jaws < teeth < lips (bullish alignment) and price > lips.
In bear 1w regime (price < 1w EMA50): go short when Alligator jaws > teeth > lips (bearish alignment) and price < lips.
Requires volume > 1.5x 20-period average for confirmation. Exit on opposite Alligator alignment or 1w regime shift.
Position size: 0.25. Target: 50-100 total trades over 4 years = 12-25/year.
Uses 1w HTF for more stable regime alignment than 1d, improving performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d data
    # Alligator: Jaw (13-period SMMA, 8 bars offset), Teeth (8-period SMMA, 5 bars offset), Lips (5-period SMMA, 3 bars offset)
    close_1d = df_1d['close'].values
    
    # SMMA (Smoothed Moving Average) implementation
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Close) / Period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close_1d, 13)  # 13-period SMMA
    teeth = smma(close_1d, 8)  # 8-period SMMA
    lips = smma(close_1d, 5)   # 5-period SMMA
    
    # Apply offsets: Jaw shifted 8 bars, Teeth shifted 5 bars, Lips shifted 3 bars
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set NaN for rolled values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align 1d Alligator lines to 1d prices (no shift needed as we're working on 1d)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Get 1w data for HTF regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for regime filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period average volume for confirmation (on 1d data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (13) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Map 1d index to 1d array index (since we're using 1d data)
        # For 1d timeframe, prices index corresponds to 1d bars
        idx_1d = i  # 1-to-1 mapping for 1d timeframe
        
        # Skip if data not ready
        if (np.isnan(jaw_aligned[idx_1d]) or 
            np.isnan(teeth_aligned[idx_1d]) or
            np.isnan(lips_aligned[idx_1d]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF regime (bullish = price above 1w EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Alligator alignments
        jaw_val = jaw_aligned[idx_1d]
        teeth_val = teeth_aligned[idx_1d]
        lips_val = lips_aligned[idx_1d]
        close_val = close[i]
        
        bullish_alignment = (jaw_val < teeth_val) and (teeth_val < lips_val)
        bearish_alignment = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        if position == 0:
            # Long setup: bullish Alligator alignment + price > lips + 1w bullish regime + volume confirmation
            long_setup = bullish_alignment and (close_val > lips_val) and htf_1w_bullish and volume_confirm
            
            # Short setup: bearish Alligator alignment + price < lips + 1w bearish regime + volume confirmation
            short_setup = bearish_alignment and (close_val < lips_val) and htf_1w_bearish and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: bearish Alligator alignment OR price < lips OR 1w regime turns bearish
            if bearish_alignment or (close_val < lips_val) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: bullish Alligator alignment OR price > lips OR 1w regime turns bullish
            if bullish_alignment or (close_val > lips_val) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Williams_Alligator_1wTrend_HTFRegime_v1"
timeframe = "1d"
leverage = 1.0