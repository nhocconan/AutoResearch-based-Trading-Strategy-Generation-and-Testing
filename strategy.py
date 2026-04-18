#!/usr/bin/env python3
"""
1h_4hTrend_1dVWAP_Reversal
Hypothesis: Use 4h EMA trend direction and 1d VWAP mean reversion for entries on 1h.
In bull markets: go long when 4h EMA21 up and price pulls back to 1d VWAP.
In bear markets: go short when 4h EMA21 down and price bounces to 1d VWAP.
Volume > 1.5x average confirms institutional interest.
Session filter (08-20 UTC) reduces noise.
Targets 20-40 trades/year by requiring EMA trend + VWAP touch + volume.
Works in both regimes by following 4h trend and fading to VWAP.
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
    
    # Pre-compute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA21 for trend
    ema_period = 21
    ema_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= ema_period:
        ema_4h[ema_period-1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2/(ema_period+1)) + (ema_4h[i-1] * (1 - 2/(ema_period+1)))
    
    # Align EMA to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d VWAP (typical price * volume / cumulative volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = np.full_like(close_1d, np.nan)
    cum_vol = 0.0
    cum_tpv = 0.0
    for i in range(len(close_1d)):
        cum_vol += volume_1d[i]
        cum_tpv += typical_price_1d[i] * volume_1d[i]
        if cum_vol > 0:
            vwap_1d[i] = cum_tpv / cum_vol
    
    # Align VWAP to 1h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period)  # EMA needs ~30 periods, vol MA needs 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: 4h EMA up + price near VWAP (below) + volume
            if i > 0 and not np.isnan(ema_4h_aligned[i-1]) and ema_4h_aligned[i] > ema_4h_aligned[i-1] and close[i] <= vwap_1d_aligned[i] * 1.005 and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: 4h EMA down + price near VWAP (above) + volume
            elif i > 0 and not np.isnan(ema_4h_aligned[i-1]) and ema_4h_aligned[i] < ema_4h_aligned[i-1] and close[i] >= vwap_1d_aligned[i] * 0.995 and vol_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: EMA down or price > VWAP*1.01
            if (i > 0 and not np.isnan(ema_4h_aligned[i-1]) and ema_4h_aligned[i] < ema_4h_aligned[i-1]) or close[i] > vwap_1d_aligned[i] * 1.01:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: EMA up or price < VWAP*0.99
            if (i > 0 and not np.isnan(ema_4h_aligned[i-1]) and ema_4h_aligned[i] > ema_4h_aligned[i-1]) or close[i] < vwap_1d_aligned[i] * 0.99:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hTrend_1dVWAP_Reversal"
timeframe = "1h"
leverage = 1.0