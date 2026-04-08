#!/usr/bin/env python3
# 4h_vwap_reversion_with_volatility_filter
# Hypothesis: In 4h timeframe, price tends to revert to VWAP during high volatility periods.
# Long when price crosses below VWAP with volatility > 1.5x average, short when above VWAP with volatility > 1.5x average.
# Exit when price crosses back above/below VWAP. Works in both bull and bear markets by capturing mean reversion during volatile swings.
# Target: 80-120 total trades over 4 years (~20-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_vwap_reversion_with_volatility_filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_tpv, cum_vol, out=np.full_like(cum_tpv, np.nan), where=cum_vol!=0)
    
    # Calculate volatility (ATR-like using true range)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    avg_atr = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup for ATR
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(avg_atr[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        volatility_high = atr[i] > 1.5 * avg_atr[i]
        
        if position == 1:  # Long position
            # Exit: price crosses back above VWAP
            if close[i] > vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back below VWAP
            if close[i] < vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion entries: price below VWAP (long) or above VWAP (short) with high volatility
            if (close[i] < vwap[i]) and volatility_high:
                position = 1
                signals[i] = 0.25
            elif (close[i] > vwap[i]) and volatility_high:
                position = -1
                signals[i] = -0.25
    
    return signals