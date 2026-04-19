#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1-day volume confirmation.
# Long when: BB width < 20th percentile, price breaks above upper BB, volume > 1.5x avg
# Short when: BB width < 20th percentile, price breaks below lower BB, volume > 1.5x avg
# Exit when price returns to middle BB or opposite band is touched.
# Designed for ~20-40 trades/year per symbol. Works in both bull and bear by trading volatility breakouts in low-vol regimes.
name = "6h_BB_Squeeze_Volume_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for BB calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands on daily close (20-period, 2 std dev)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Calculate 50th percentile of BB width for threshold (squeeze = low volatility)
    # Use expanding window to avoid look-ahead
    bb_width_pct = np.full_like(bb_width, np.nan)
    for i in range(20, len(bb_width)):  # Need at least 20 values for percentile
        if i >= 20:
            bb_width_pct[i] = pd.Series(bb_width[:i+1]).rolling(window=50, min_periods=10).quantile(0.2).values[-1] if i >= 50 else np.percentile(bb_width[:i+1], 20)
    
    # Align BB levels and squeeze signal to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)
    
    # 20-period SMA on 6h data for exit
    sma_20_6h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(bb_width_pct_aligned[i]) or np.isnan(sma_20_6h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_width = upper_bb_aligned[i] - lower_bb_aligned[i]
        bb_width_threshold = bb_width_pct_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Check for BB squeeze (low volatility)
            if bb_width <= bb_width_threshold and vol > 1.5 * vol_ma:
                # Long breakout: price breaks above upper BB
                if price > upper_bb_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: price breaks below lower BB
                elif price < lower_bb_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to middle BB or touches lower BB
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if price <= middle_bb or price < lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle BB or touches upper BB
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if price >= middle_bb or price > upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals