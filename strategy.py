#!/usr/bin/env python3
"""
4h Volume-Weighted Price Action with 12h Trend Filter
Hypothesis: In both bull and bear markets, price moves with institutional volume
show persistence. We use 12h EMA trend filter to avoid counter-trend trades,
and enter on 4h when price deviates significantly from VWAP with volume confirmation.
This strategy targets 20-30 trades/year to minimize fee drag while capturing
strong momentum moves. VWAP deviation identifies overextended moves likely to
reverse or continue with volume, while 12h trend ensures we trade with the
dominant higher timeframe momentum.
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
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate VWAP for 4h
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Cumulative VWAP with reset at session start (daily)
    # We'll use rolling window as proxy for session VWAP
    vwap = pd.Series(vwap_numerator).rolling(window=28, min_periods=14).sum().values / \
           pd.Series(vwap_denominator).rolling(window=28, min_periods=14).sum().values
    
    # VWAP deviation in ATR units
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    vwap_dev = (close - vwap) / atr  # Deviation in ATR multiples
    
    # Volume filter: current volume > 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vwap_dev[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vdev = vwap_dev[i]
        vol_ok = vol_filter[i]
        trend = ema34_12h_aligned[i]
        
        if position == 0:
            # Long: price above VWAP with volume, in uptrend
            if vdev > 0.8 and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP with volume, in downtrend
            elif vdev < -0.8 and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to VWAP or trend weakens
            if vdev < 0.2 or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to VWAP or trend weakens
            if vdev > -0.2 or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Deviation_Volume_12hTrend"
timeframe = "4h"
leverage = 1.0