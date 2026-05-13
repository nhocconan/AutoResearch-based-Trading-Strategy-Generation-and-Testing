#!/usr/bin/env python3
"""
4h_VWAP_Reversion_With_Volume_Spike
Hypothesis: Price reverts to VWAP on 4h timeframe during low volatility periods 
when volume spikes indicate exhaustion of short-term momentum. Works in both 
bull and bear markets as mean reversion strategy with volatility filter.
"""

name = "4h_VWAP_Reversion_With_Volume_Spike"
timeframe = "4h"
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
    
    # Calculate VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # Price deviation from VWAP as percentage
    price_dev = (close - vwap) / vwap
    
    # Volatility filter: ATR(14) / SMA(close, 50) - low volatility regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    vol_ratio = atr / sma50
    low_vol = vol_ratio < 0.02  # Low volatility threshold
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        if low_vol[i] and volume_spike[i]:
            # Mean reversion: price deviated from VWAP
            if price_dev[i] > 0.015:  # Overbought - short
                signals[i] = -0.25
            elif price_dev[i] < -0.015:  # Oversold - long
                signals[i] = 0.25
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals