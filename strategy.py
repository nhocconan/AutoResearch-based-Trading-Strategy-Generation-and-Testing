#!/usr/bin/env python3
"""
4h_KAMA_Trend_1dVWAP_MeanReversion
Hypothesis: KAMA identifies trend on 4h; mean-reversion to 1d VWAP during pullbacks.
Works in bull via trend continuation, in bear via reversion to mean during pullbacks.
Target: 15-30 trades/year (60-120 total) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_ktf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate VWAP on 1d: cumulative (price * volume) / cumulative volume
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    pv_1d = typical_price_1d * df_1d['volume'].values
    cum_pv = np.cumsum(pv_1d)
    cum_vol = np.cumsum(df_1d['volume'].values)
    vwap_1d = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Align 1d VWAP to 4h
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # KAMA on 4h close
    def kama(close, period=10, fast=2, slow=30):
        dir = np.abs(np.diff(close, period))
        vol = np.sum(np.abs(np.diff(close)), axis=1) if len(close) > 1 else np.array([1e-10])
        er = np.where(vol != 0, dir / vol, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_out = np.full_like(close, np.nan)
        kama_out[period] = close[period]
        for i in range(period+1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_val = kama(close, 10, 2, 30)
    
    # Pullback detection: price deviation from VWAP
    dev_pct = (close - vwap_aligned) / vwap_aligned
    
    # Volume filter: avoid low-volume whipsaws
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    start_idx = max(100, 20)  # KAMA needs warmup, vol MA needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(kama_val[i]) or
            np.isnan(vwap_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend: price vs KAMA
        uptrend = price > kama_val[i]
        downtrend = price < kama_val[i]
        
        # Mean-reversion signal: deviation from VWAP
        if position == 0:
            # Long: uptrend + price significantly below VWAP (pullback to buy)
            if uptrend and dev_pct[i] < -0.015 and vol_ratio > 1.2:
                signals[i] = size
                position = 1
            # Short: downtrend + price significantly above VWAP (pullback to sell)
            elif downtrend and dev_pct[i] > 0.015 and vol_ratio > 1.2:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or trend breaks
            if dev_pct[i] > -0.005 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to VWAP or trend breaks
            if dev_pct[i] < 0.005 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_1dVWAP_MeanReversion"
timeframe = "4h"
leverage = 1.0