#!/usr/bin/env python3
"""
1h Volume Spike Reversion with 4h Trend Filter
Hypothesis: After extreme volume spikes, price often reverts to the mean. 
We use 4h EMA50 as trend filter to avoid counter-trend trades, and enter on 1h 
when price deviates significantly from VWAP with volume confirmation (>2x avg volume).
This strategy targets 15-30 trades/year by requiring multiple confirmations,
reducing fee drag while capturing mean reversion moves in both bull and bear markets.
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
    
    # Get 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate VWAP for 1h
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Rolling VWAP (24 periods = 1 day)
    vwap = pd.Series(vwap_numerator).rolling(window=24, min_periods=12).sum().values / \
           pd.Series(vwap_denominator).rolling(window=24, min_periods=12).sum().values
    
    # VWAP deviation in ATR units
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    vwap_dev = (close - vwap) / atr  # Deviation in ATR multiples
    
    # Volume filter: current volume > 2x 24-period volume average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vwap_dev[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vdev = vwap_dev[i]
        vol_ok = vol_filter[i]
        sess_ok = session_filter[i]
        trend = ema50_4h_aligned[i]
        
        if position == 0:
            # Long: price below VWAP with high volume, in uptrend (mean reversion long)
            if vdev < -1.0 and vol_ok and sess_ok and price > trend:
                signals[i] = 0.20
                position = 1
            # Short: price above VWAP with high volume, in downtrend (mean reversion short)
            elif vdev > 1.0 and vol_ok and sess_ok and price < trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit if price returns to VWAP or trend weakens
            if vdev > -0.2 or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit if price returns to VWAP or trend weakens
            if vdev < 0.2 or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Volume_Spike_Reversion_4hTrend"
timeframe = "1h"
leverage = 1.0