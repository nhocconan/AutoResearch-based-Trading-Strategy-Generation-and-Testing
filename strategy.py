#!/usr/bin/env python3
"""
4h VWAP Reversion with Volume Spike and 1D Trend Filter
Hypothesis: Price reverting to VWAP after a volume spike, aligned with 1D trend (close > EMA50), captures mean reversion in strong trends. 
Works in bull (buying dips) and bear (selling rallies). Target: 20-30 trades/year to minimize fee drag.
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
    
    # VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.cumsum(tpv)
    cum_vol = np.cumsum(volume)
    vwap = cum_tpv / cum_vol
    
    # Volume spike: current volume > 2.0 x 20-period EMA of volume
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema)
    
    # 1D trend: EMA50 on daily timeframe
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(vwap[i]) or np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        trend = ema50_1d_aligned[i]
        spike = vol_spike[i]
        
        if position == 0:
            # Long: price below VWAP, volume spike, uptrend (price > EMA50)
            if price < vwap_val and spike and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP, volume spike, downtrend (price < EMA50)
            elif price > vwap_val and spike and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses above VWAP or trend weakens
            if price > vwap_val or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses below VWAP or trend weakens
            if price < vwap_val or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Reversion_VolumeSpike_1DTrend"
timeframe = "4h"
leverage = 1.0