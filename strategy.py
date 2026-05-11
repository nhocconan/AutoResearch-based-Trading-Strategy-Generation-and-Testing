#!/usr/bin/env python3
"""
6h_VolumeSpike_1dTrend_FadeFromDailyVWAP
Hypothesis: Fade price moves away from daily VWAP on 6h timeframe during low volatility periods, 
but only when aligned with daily trend. Uses volume spike as entry trigger and daily trend filter.
Works in bull/bear markets by fading mean-reversion moves in trending environments.
"""

name = "6h_VolumeSpike_1dTrend_FadeFromDailyVWAP"
timeframe = "6h"
leverage = 1.0

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
    
    # === Daily VWAP ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP components
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    pv = typical_price * df_1d['volume']
    cum_pv = pv.cumsum()
    cum_vol = df_1d['volume'].cumsum()
    vwap = cum_pv / cum_vol
    vwap_values = vwap.values
    
    # Align VWAP to 6h timeframe
    vwap_6h = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # === Daily Trend Filter (EMA50) ===
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Spike Filter (2.5x 30-period EMA on 6h) ===
    vol_ema30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_spike = volume > vol_ema30 * 2.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers daily calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_6h[i]) or np.isnan(ema50_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price below VWAP, uptrend, volume spike (fade downside deviation)
            if (close[i] < vwap_6h[i] and 
                close[i] > ema50_6h[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price above VWAP, downtrend, volume spike (fade upside deviation)
            elif (close[i] > vwap_6h[i] and 
                  close[i] < ema50_6h[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above VWAP (mean reversion complete) or trend breaks
            if close[i] >= vwap_6h[i] or close[i] < ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses below VWAP (mean reversion complete) or trend breaks
            if close[i] <= vwap_6h[i] or close[i] > ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals