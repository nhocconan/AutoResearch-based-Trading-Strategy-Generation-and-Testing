#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_HighLow_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses Camarilla pivot levels (H4/L4 as extreme breakout, H3/L3 as mean reversion) with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar MA). Designed for low trade frequency (~20-30 trades/year) by requiring confluence of HTF trend, extreme price level breach, and volume spike. Works in bull/bear markets by following 1d trend while using Camarilla structure for precise mean-reversion exits at H3/L3 levels. Volume filter reduces false breakouts. Prioritizes BTC/ETH with SOL as secondary.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous day's OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: H4, H3, L3, L4
    rng = high_1d - low_1d
    camarilla_h4 = close_1d_vals + (rng * 1.1 / 2)   # H4 = same as R1/R2 extreme
    camarilla_h3 = close_1d_vals + (rng * 1.1 / 4)   # H3
    camarilla_l3 = close_1d_vals - (rng * 1.1 / 4)   # L3
    camarilla_l4 = close_1d_vals - (rng * 1.1 / 2)   # L4 = same as S1/S2 extreme
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 1.5x 20-period average (dynamic threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for vol, 50 for 1d EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        camarilla_h4_val = camarilla_h4_aligned[i]
        camarilla_h3_val = camarilla_h3_aligned[i]
        camarilla_l3_val = camarilla_l3_aligned[i]
        camarilla_l4_val = camarilla_l4_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Entry conditions: breakout of Camarilla H4/L4 in trend direction with volume spike
        long_entry = (close_val > camarilla_h4_val) and bullish_1d and vol_spike
        short_entry = (close_val < camarilla_l4_val) and bearish_1d and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to H3 (profit target) or trend change
            if close_val < camarilla_h3_val or not bullish_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion to L3 (profit target) or trend change
            if close_val > camarilla_l3_val or not bearish_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_Pivot_HighLow_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0