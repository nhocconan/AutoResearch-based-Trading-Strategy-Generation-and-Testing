#!/usr/bin/env python3
"""
6h_VWAP_Deviation_1dTrend_Filter
Hypothesis: Price reverts to VWAP during institutional hours. Go long when price deviates below VWAP 
with 1d uptrend and volume confirmation. Go short when price deviates above VWAP with 1d downtrend 
and volume confirmation. Uses 6h timeframe to capture institutional reversion moves, avoiding 
noise from lower timeframes. VWAP calculated using typical price * volume.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
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
    
    # Calculate typical price and VWAP components
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    
    # Cumulative VWAP (reset daily)
    cum_tp_vol = np.zeros(n)
    cum_vol = np.zeros(n)
    vwap = np.zeros(n)
    
    # Get date for daily reset
    dates = pd.to_datetime(prices['open_time']).date
    
    for i in range(n):
        if i == 0 or dates[i] != dates[i-1]:
            cum_tp_vol[i] = tp_vol[i]
            cum_vol[i] = volume[i]
        else:
            cum_tp_vol[i] = cum_tp_vol[i-1] + tp_vol[i]
            cum_vol[i] = cum_vol[i-1] + volume[i]
        
        if cum_vol[i] > 0:
            vwap[i] = cum_tp_vol[i] / cum_vol[i]
        else:
            vwap[i] = typical_price[i]
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    uptrend = close > ema_20_1d_aligned
    downtrend = close < ema_20_1d_aligned
    
    # Volume confirmation: volume > 1.3x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.3)
    
    # VWAP deviation bands (1.5% deviation)
    vwap_upper = vwap * 1.015
    vwap_lower = vwap * 0.985
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap[i]) or np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: Price below VWAP lower band, with 1d uptrend and volume confirmation
            if close[i] < vwap_lower[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price above VWAP upper band, with 1d downtrend and volume confirmation
            elif close[i] > vwap_upper[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price crosses above VWAP OR 1d trend changes to downtrend
            if close[i] > vwap[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price crosses below VWAP OR 1d trend changes to uptrend
            if close[i] < vwap[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_VWAP_Deviation_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0