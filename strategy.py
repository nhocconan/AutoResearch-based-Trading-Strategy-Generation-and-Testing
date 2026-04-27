#!/usr/bin/env python3
"""
6h_Aroon_Oscillator_Trend_1wTrend_Filter
Hypothesis: Aroon Oscillator (25) identifies strong trends on 6h timeframe. 
Only trade in direction of 1-week trend (price above/below 1w EMA50) to avoid counter-trend trades.
Add volume confirmation (volume > 1.5x 20-period average) to filter false breakouts.
Target: 60-120 total trades over 4 years (15-30/year) with discrete sizing (0.25).
Works in bull markets via trend continuation and in bear markets via trend-following shorts.
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
    
    # Aroon Oscillator (25-period) on 6h data
    # Aroon Up = ((period - days since highest high) / period) * 100
    # Aroon Down = ((period - days since lowest low) / period) * 100
    # Oscillator = Aroon Up - Aroon Down (range -100 to +100)
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period, n):
        window_high = high[i-period+1:i+1]
        window_low = low[i-period+1:i+1]
        highest_high_idx = np.argmax(window_high)
        lowest_low_idx = np.argmin(window_low)
        periods_since_high = period - 1 - highest_high_idx
        periods_since_low = period - 1 - lowest_low_idx
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    aroon_osc = aroon_up - aroon_down  # -100 to +100
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all indicators to primary timeframe (6h)
    aroon_osc_aligned = aroon_osc  # Already calculated on 6h data
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm)  # volume is LTF, but confirm using 1w avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Aroon (25), EMA50 (50), volume avg (20)
    start_idx = max(25, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(aroon_osc_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        aroon = aroon_osc_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs EMA50 (1w)
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            # Long: Aroon Oscillator > 50 (strong uptrend) + volume + uptrend
            if uptrend and aroon > 50 and vol_conf:
                signals[i] = size
                position = 1
            # Short: Aroon Oscillator < -50 (strong downtrend) + volume + downtrend
            elif downtrend and aroon < -50 and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: Aroon Oscillator < 0 (trend weakening) or reverse signal
            if aroon < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: Aroon Oscillator > 0 (trend weakening) or reverse signal
            if aroon > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Aroon_Oscillator_Trend_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0