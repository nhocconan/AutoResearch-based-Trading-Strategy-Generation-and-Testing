#!/usr/bin/env python3
name = "6h_WeeklyPivot_Momentum_Breakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly pivot points (using previous week)
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    prev_close_w = df_1w['close'].shift(1).values
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    r1_w = 2 * pivot_w - prev_low_w
    s1_w = 2 * pivot_w - prev_high_w
    
    # Align weekly pivots to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # Get daily data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily RSI(14) for momentum
    delta = np.diff(df_1d['close'].values, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_loss / (avg_gain + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Volume filter: current volume > 1.5x 24-period average (6 days for 6h)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 6h to reduce trades
    
    start_idx = max(100, 24, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_w_aligned[i]) or 
            np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above weekly R1 with bullish momentum (RSI > 50)
            if (close[i] > r1_w_aligned[i] and 
                rsi_14_aligned[i] > 50 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below weekly S1 with bearish momentum (RSI < 50)
            elif (close[i] < s1_w_aligned[i] and 
                  rsi_14_aligned[i] < 50 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below weekly pivot or momentum turns bearish
            if close[i] < pivot_w_aligned[i] or rsi_14_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above weekly pivot or momentum turns bullish
            if close[i] > pivot_w_aligned[i] or rsi_14_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot points provide strong support/resistance levels that work across market regimes.
# In bull markets: buy breakouts above weekly R1 with bullish momentum (RSI>50).
# In bear markets: sell breakdowns below weekly S1 with bearish momentum (RSI<50).
# Uses 6h timeframe for lower trade frequency (~15-30 trades/year) to minimize fee drag.
# Weekly pivot avoids noise of daily pivots; RSI filters false breakouts.