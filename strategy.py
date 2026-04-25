#!/usr/bin/env python3
"""
12h Camarilla Pivot Breakout with Volume Spike and ATR Filter
Hypothesis: Camarilla R1/S1 breakouts with volume confirmation and ATR-based volatility filter
capture momentum moves in both bull and bear markets. The ATR filter ensures we only trade
when volatility is sufficient for meaningful breakouts, reducing whipsaws. Uses discrete
position sizing (0.25) to minimize fee churn and targets 12-30 trades/year on 12h.
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
    
    # ATR for volatility filter (14-period)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Volatility filter: ATR > 0.8 * 50-period ATR MA (ensures sufficient volatility)
    vol_filter = atr_14 > (atr_ma_50 * 0.8)
    
    # 1d Camarilla pivot levels (MTF)
    df_1d = get_htf_data(prices, '1d')
    camarilla_r1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    camarilla_s1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 50) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_ma_50[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        vol_ok = vol_filter[i]
        
        # Breakout conditions: price breaks Camarilla R1 or S1 levels
        breakout_long = curr_close > r1_aligned[i]
        breakout_short = curr_close < s1_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: pivot breakout + volume spike + volatility filter
            long_entry = breakout_long and vol_spike and vol_ok
            short_entry = breakout_short and vol_spike and vol_ok
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on close below S1 (mean reversion) or opposite breakout
            if curr_close < s1_aligned[i] or breakout_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on close above R1 (mean reversion) or opposite breakout
            if curr_close > r1_aligned[i] or breakout_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_VolumeSpike_VolFilter"
timeframe = "12h"
leverage = 1.0