#!/usr/bin/env python3
"""
4h_VWAP_Breakout_12hTrend
Hypothesis: On 4h timeframe, price breaking above/below VWAP with 12h EMA34 trend confirmation and volume surge captures institutional flow. 
VWAP acts as dynamic support/resistance; breakouts with volume indicate strong momentum. 
12h EMA34 filters for higher timeframe trend to avoid counter-trend trades. 
Designed for 20-35 trades/year with clear entry/exit rules to minimize whipsaw and fee impact.
Works in bull/bear by following higher timeframe trend direction only.
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
    
    # VWAP calculation (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = np.where(cum_vol != 0, cum_tpv / cum_vol, 0.0)
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema34_12h[33] = np.mean(close_12h[:34])
        k = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema34_12h[i] = close_12h[i] * k + ema34_12h[i-1] * (1 - k)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    vol_surge = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for VWAP and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(vwap[i]) or np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP with uptrend and volume surge
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1] and ema34_12h_aligned[i] > ema34_12h_aligned[i-1] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP with downtrend and volume surge
            elif close[i] < vwap[i] and close[i-1] >= vwap[i-1] and ema34_12h_aligned[i] < ema34_12h_aligned[i-1] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below VWAP or trend turns down
            if close[i] < vwap[i] or ema34_12h_aligned[i] <= ema34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above VWAP or trend turns up
            if close[i] > vwap[i] or ema34_12h_aligned[i] >= ema34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Breakout_12hTrend"
timeframe = "4h"
leverage = 1.0