#!/usr/bin/env python3
"""
6h_WilliamsVixFix_MeanReversion
Hypothesis: Williams Vix Fix (WVF) identifies volatility spikes and mean reversion opportunities on 6h timeframe.
Uses 12h EMA50 as trend filter to avoid counter-trend trades. Works in bull/bear via adaptive mean reversion.
Target: 15-25 trades/year per symbol (~60-100 total over 4 years) to minimize fee drag.
"""

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
    
    # Get 6h data for indicators
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams Vix Fix on 6h
    # WVF = ((Highest Close in period - Low) / (Highest Close in period)) * 100
    # We invert it to get a volatility measure: higher WVF = higher fear/volatility
    highest_close_22 = pd.Series(close).rolling(window=22, min_periods=22).max().values
    wvf = ((highest_close_22 - low) / highest_close_22) * 100
    
    # Smooth WVF with EMA10 for signal line
    wvf_smooth = pd.Series(wvf).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Calculate mean reversion z-score on 6h close
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    zscore = (close - sma_20) / (std_20 + 1e-9)  # Avoid division by zero
    
    # Align all indicators
    wvf_smooth_aligned = wvf_smooth
    zscore_aligned = zscore
    ema_50_12h_aligned = ema_50_12h_aligned  # Already aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need WVF (22), zscore (20), EMA50 (50)
    start_idx = max(50, 22, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(wvf_smooth_aligned[i]) or 
            np.isnan(zscore_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Mean reversion conditions:
        # Long: High volatility (WVF > 80) + oversold (z-score < -1.5) + price below EMA (discount in uptrend)
        # Short: High volatility (WVF > 80) + overbought (z-score > 1.5) + price above EMA (premium in downtrend)
        long_condition = (wvf_smooth_aligned[i] > 80 and 
                         zscore_aligned[i] < -1.5 and 
                         price_below_ema)
        short_condition = (wvf_smooth_aligned[i] > 80 and 
                          zscore_aligned[i] > 1.5 and 
                          price_above_ema)
        
        if position == 0:
            # Enter long on mean reversion long signal
            if long_condition:
                signals[i] = 0.25
                position = 1
            # Enter short on mean reversion short signal
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: volatility decreases OR mean reversion complete OR trend changes
            if (wvf_smooth_aligned[i] < 40 or  # Low volatility
                zscore_aligned[i] > -0.5 or    # Mean reversion complete
                not price_below_ema):          # Trend changed
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: volatility decreases OR mean reversion complete OR trend changes
            if (wvf_smooth_aligned[i] < 40 or   # Low volatility
                zscore_aligned[i] < 0.5 or      # Mean reversion complete
                not price_above_ema):           # Trend changed
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVixFix_MeanReversion"
timeframe = "6h"
leverage = 1.0