#!/usr/bin/env python3
"""
12h Volume-Weighted Price Action with Daily Trend Filter and ATR Stop
Hypothesis: Combines volume confirmation with price action near VWAP and daily EMA trend
to capture sustainable moves. Designed for 12-37 trades/year on 12h timeframe.
Works in bull markets via trend continuation and bear markets via mean reversion
at VWAP with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop
    df_d = get_htf_data(prices, '1d')
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(df_d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_d, ema_50)
    
    # VWAP calculation (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # Volume spike: 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Price deviation from VWAP (normalized by ATR for stability)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    vwap_dev = (close - vwap) / atr  # Deviation in ATR units
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or 
            np.isnan(vwap_dev[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema = ema_aligned[i]
        atr_val = atr[i]
        dev = vwap_dev[i]
        
        if position == 0:
            # Long: price below VWAP (mean reversion) in uptrend with volume spike
            if dev < -0.8 and volume_spike[i] and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP (mean reversion) in downtrend with volume spike
            elif dev > 0.8 and volume_spike[i] and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price crosses VWAP or ATR trailing stop
            if dev > 0.2 or price < (high[i] - 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price crosses VWAP or ATR trailing stop
            if dev < -0.2 or price > (low[i] + 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_VWAP_MeanReversion_VolumeSpike_EMA50"
timeframe = "12h"
leverage = 1.0