#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Intraday Range with Daily Opening Gap Fill
# Hypothesis: Price tends to fill overnight gaps (previous day's open) during 6h sessions
# with institutional participation. Gap fill acts as mean reversion in ranging markets
# and continuation in trending markets when combined with volume confirmation.
# Works in bull/bear via mean reversion at key levels. Target: 20-40 trades/year.
name = "6h_gap_fill_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get daily data for gap calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's open (gap level)
    prev_open = df_1d['open'].shift(1).values
    
    # Align daily open to 6h timeframe
    prev_open_6h = align_htf_to_ltf(prices, df_1d, prev_open)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price action: close within 0.5% of gap level indicates fill attempt
    gap_threshold = 0.005  # 0.5%
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(prev_open_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Distance to gap level (as fraction of price)
        gap_dist = abs(close[i] - prev_open_6h[i]) / close[i]
        near_gap = gap_dist <= gap_threshold
        
        if position == 1:  # Long position
            # Exit: price moves away from gap or volume dries up
            if gap_dist > gap_threshold * 2 or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price moves away from gap or volume dries up
            if gap_dist > gap_threshold * 2 or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry at gap
            # Enter long: price below gap, moving up to fill it with volume
            if close[i] < prev_open_6h[i] and near_gap and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price above gap, moving down to fill it with volume
            elif close[i] > prev_open_6h[i] and near_gap and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals