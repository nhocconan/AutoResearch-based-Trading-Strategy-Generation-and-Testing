#!/usr/bin/env python3
"""
6h Time-Weighted Average Price (TWAP) Reversion with 1d Trend Filter
Hypothesis: Price reverts to its time-weighted average (TWAP) on the 6h timeframe,
but only when aligned with the daily trend. In bull markets, we buy dips to TWAP
in uptrends; in bear markets, we sell rallies to TWAP in downtrends. This
mean-reversion strategy works in both regimes by following the higher timeframe
trend, reducing false signals. Uses volume confirmation to avoid low-liquidity
noise. Targets 15-25 trades/year per symbol to minimize fee drag.
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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate TWAP (typical price weighted by time, approximated by volume)
    typical_price = (high + low + close) / 3.0
    twap_numerator = typical_price * volume
    twap_denominator = volume
    
    # Rolling TWAP (24 periods = 6h * 24 = 6 days)
    twap = pd.Series(twap_numerator).rolling(window=24, min_periods=12).sum().values / \
           pd.Series(twap_denominator).rolling(window=24, min_periods=12).sum().values
    
    # TWAP deviation in ATR units
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    twap_dev = (close - twap) / atr  # Deviation in ATR multiples
    
    # Volume filter: current volume > 1.3x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(twap_dev[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tdev = twap_dev[i]
        vol_ok = vol_filter[i]
        trend = ema50_1d_aligned[i]
        
        if position == 0:
            # Long: price below TWAP with volume, in uptrend (buy the dip)
            if tdev < -0.6 and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price above TWAP with volume, in downtrend (sell the rally)
            elif tdev > 0.6 and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to TWAP or trend weakens
            if tdev > -0.2 or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to TWAP or trend weakens
            if tdev < 0.2 or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_TWAP_Reversion_1dTrend"
timeframe = "6h"
leverage = 1.0