#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly Keltner channel and daily volume confirmation.
# Uses weekly EMA-based Keltner channels (EMA20 ± 2*ATR) for trend and breakout signals.
# Fades at channel midlines in direction of weekly trend and breaks out at channel extremes.
# Volume filter confirms institutional participation. Designed for 15-35 trades/year on 6h.
# Weekly trend filter reduces whipsaw in sideways markets and improves win rate in both bull and bear regimes.

name = "6h_1w_keltner_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly ATR(10)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_shift = np.roll(close_1w, 1)
    close_1w_shift[0] = close_1w[0]
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w_shift)
    tr3 = np.abs(low_1w - close_1w_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Weekly Keltner channels
    upper_1w = ema20_1w + 2 * atr10_1w
    lower_1w = ema20_1w - 2 * atr10_1w
    midline_1w = ema20_1w
    
    # Align weekly channels to 6h
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    midline_1w_aligned = align_htf_to_ltf(prices, df_1w, midline_1w)
    
    # Daily average volume (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(midline_1w_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * daily average volume
        vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Weekly trend: price above/below midline
        price_above_mid = close[i] > midline_1w_aligned[i]
        price_below_mid = close[i] < midline_1w_aligned[i]
        
        # Fade at midline in direction of weekly trend
        fade_long = (low[i] <= midline_1w_aligned[i] and vol_filter and price_above_mid)
        fade_short = (high[i] >= midline_1w_aligned[i] and vol_filter and price_below_mid)
        
        # Breakout at channel extremes
        breakout_long = (high[i] >= upper_1w_aligned[i] and vol_filter)
        breakout_short = (low[i] <= lower_1w_aligned[i] and vol_filter)
        
        # Exit when price returns to opposite channel extreme or midline
        exit_long = (position == 1 and 
                    (low[i] <= lower_1w_aligned[i] or  # Hit opposite extreme
                     high[i] >= upper_1w_aligned[i]))  # Hit same extreme (take profit)
        exit_short = (position == -1 and 
                     (high[i] >= upper_1w_aligned[i] or  # Hit opposite extreme
                      low[i] <= lower_1w_aligned[i]))   # Hit same extreme (take profit)
        
        # Priority: breakout > fade > hold
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif fade_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif fade_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals