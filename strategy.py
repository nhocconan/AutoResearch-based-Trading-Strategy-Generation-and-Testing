#!/usr/bin/env python3
"""
1d_VWAP_Pullback_1wTrend_Volume
Hypothesis: Pullbacks to VWAP on strong weekly trends with volume confirmation capture mean-reversion in trending markets. Works in bull by buying dips, in bear by selling rallies, using 1-week EMA50 trend filter and volume spike (1.5x average) to filter noise. Targets 50-80 total trades over 4 years to minimize fee drag.
"""
name = "1d_VWAP_Pullback_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # VWAP calculation (typical price * volume) cumulative
    typical_price = (high + low + close) / 3
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient warmup for averages
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price pulls back to VWAP in uptrend + volume confirmation
            if close[i] <= vwap[i] and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price rallies to VWAP in downtrend + volume confirmation
            elif close[i] >= vwap[i] and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price moves away from VWAP by 1% or opposite VWAP touch
            if position == 1:
                if close[i] >= vwap[i] * 1.01 or close[i] <= vwap[i] * 0.99:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] <= vwap[i] * 0.99 or close[i] >= vwap[i] * 1.01:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals