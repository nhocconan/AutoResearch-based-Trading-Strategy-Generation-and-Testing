#!/usr/bin/env python3
"""
4h_VWAP_Breakout_1dTrend_Volume
Hypothesis: Price breaking above/below VWAP with 1d trend filter and volume confirmation captures institutional flow.
VWAP acts as dynamic support/resistance. In trending markets, price tends to stay above/below VWAP with institutional participation.
Volume surge confirms smart money involvement. Works in bull (buying above VWAP) and bear (selling below VWAP).
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "4h_VWAP_Breakout_1dTrend_Volume"
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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Calculate VWAP for 4h data
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(vwap[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled to 4h)
        # Approximate 4h volume from 1d: 1d volume / 6 (since 24h/4h = 6)
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: Price crosses above VWAP with uptrend and volume confirmation
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below VWAP with downtrend and volume confirmation
            elif close[i] < vwap[i] and close[i-1] >= vwap[i-1] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses below VWAP or trend reversal
            if close[i] < vwap[i] and close[i-1] >= vwap[i-1] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above VWAP or trend reversal
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals