#!/usr/bin/env python3
"""
4h_Bollinger_Width_Regime_Adaptive_V1
Hypothesis: Market regime detection using Bollinger Band width percentile (low = range, high = trend) 
combined with mean-reversion in range and trend-following in trending regimes. 
Uses Bollinger Bands for entries and exits with volume confirmation. Designed for low trade frequency 
(20-40/year) to minimize fee drag and work in both bull and bear markets via regime adaptation.
"""

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
    
    # Bollinger Bands (20, 2.0)
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma20 + (2.0 * std20)
    lower_bb = ma20 - (2.0 * std20)
    bb_width = (upper_bb - lower_bb) / ma20  # Normalized width
    
    # Bollinger Width Percentile (50-period) for regime detection
    bb_width_series = pd.Series(bb_width)
    bw_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need BB (20), BW percentile (50), volume avg (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ma20[i]) or np.isnan(std20[i]) or np.isnan(bw_percentile[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bw = bw_percentile[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Regime detection: bw < 30 = range, bw > 70 = trend
            if bw < 30:  # Range regime - mean reversion
                if close_val <= lower_bb and vol_conf:
                    signals[i] = size  # Long at lower band
                    position = 1
                elif close_val >= upper_bb and vol_conf:
                    signals[i] = -size  # Short at upper band
                    position = -1
            elif bw > 70:  # Trend regime - trend following
                # Simple trend: price above/below MA20
                if close_val > ma20[i] and vol_conf:
                    signals[i] = size  # Long in uptrend
                    position = 1
                elif close_val < ma20[i] and vol_conf:
                    signals[i] = -size  # Short in downtrend
                    position = -1
        elif position == 1:
            # Exit conditions
            if bw < 30:  # Range: exit at opposite band or middle
                if close_val >= ma20[i]:
                    signals[i] = 0.0
                    position = 0
            else:  # Trend: exit on trend reversal or volatility contraction
                if close_val < ma20[i] or bw < 50:
                    signals[i] = 0.0
                    position = 0
            if position == 1:
                signals[i] = size
        elif position == -1:
            # Exit conditions
            if bw < 30:  # Range: exit at opposite band or middle
                if close_val <= ma20[i]:
                    signals[i] = 0.0
                    position = 0
            else:  # Trend: exit on trend reversal or volatility contraction
                if close_val > ma20[i] or bw < 50:
                    signals[i] = 0.0
                    position = 0
            if position == -1:
                signals[i] = -size
    
    return signals

name = "4h_Bollinger_Width_Regime_Adaptive_V1"
timeframe = "4h"
leverage = 1.0