#!/usr/bin/env python3
"""
4h_VWAP_MeanReversion_TrendFilter_Volume
Hypothesis: Price reverts to VWAP during strong trends (1d EMA50) with volume confirmation (1.5x average) to capture mean-reversion in trending markets. Works in both bull and bear markets by following higher timeframe trend while exploiting short-term deviations from VWAP. Designed for 4h timeframe to balance trade frequency and edge.
"""
name = "4h_VWAP_MeanReversion_TrendFilter_Volume"
timeframe = "4h"
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
    
    # Calculate VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3
    tpv = typical_price * volume
    cum_tpv = np.cumsum(tpv)
    cum_vol = np.cumsum(volume)
    vwap = cum_tpv / cum_vol
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient warmup for averages
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(vwap[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below VWAP + 1d uptrend + volume surge
            if close[i] < vwap[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP + 1d downtrend + volume surge
            elif close[i] > vwap[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to VWAP (mean reversion)
            if position == 1:
                if close[i] >= vwap[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] <= vwap[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals