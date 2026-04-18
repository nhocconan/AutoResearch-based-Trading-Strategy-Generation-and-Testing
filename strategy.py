#!/usr/bin/env python3
"""
4h_BollingerBandwidth_Breakout_Volume_Trend
Hypothesis: Breakouts from Bollinger Bandwidth extremes with volume confirmation and EMA trend filter work in both bull and bear markets.
In low volatility (bandwidth < 20th percentile), price often breaks out with momentum. High bandwidth (>80th) signals exhaustion.
Uses Bollinger Bands (20,2) to define dynamic channels. Requires volume > 1.3x 20-period average and EMA20 alignment.
Targets 20-30 trades/year (80-120 total) to avoid fee drag while capturing volatility expansion moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Bandwidth = (Upper - Lower) / Middle
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Percentile lookback for regime (252 days ~ 252*6=1512 bars for 4h)
    lookback = min(1512, i if 'i' in locals() else 1512)  # Will be computed in loop
    # We'll compute percentile in loop to avoid look-ahead
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    # EMA20 trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for BB and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(bb_width[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        vol_ok = volume_filter[i]
        ema20 = ema_20[i]
        
        # Calculate bandwidth percentile lookback (up to 252 days)
        lb_start = max(0, i - 1512)
        bb_width_slice = bb_width[lb_start:i+1]
        if len(bb_width_slice) < 20:  # Need minimum for percentile
            bandwidth_pct = 50  # neutral
        else:
            bandwidth_pct = (bb_width_slice <= bb_width[i]).sum() * 100.0 / len(bb_width_slice)
        
        if position == 0:
            # Long: breakout above upper BB in low volatility (bandwidth < 20th percentile) with volume in uptrend
            if price > upper and bandwidth_pct < 20 and vol_ok and price > ema20:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower BB in low volatility with volume in downtrend
            elif price < lower and bandwidth_pct < 20 and vol_ok and price < ema20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to middle (SMA) or volatility expands (bandwidth > 80th)
            if price < sma_20[i] or bandwidth_pct > 80:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to middle (SMA) or volatility expands (bandwidth > 80th)
            if price > sma_20[i] or bandwidth_pct > 80:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_BollingerBandwidth_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0