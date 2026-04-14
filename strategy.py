#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day ATR(14) volatility filter and volume confirmation.
# The 1-day ATR(14) filter ensures trades occur only during sufficient volatility, reducing whipsaw in low-volatility periods.
# The Donchian(20) breakout captures momentum in the direction of the breakout.
# Volume > 1.3x the 20-period average confirms participation and reduces false breakouts.
# Exit occurs when price returns to the midpoint of the Donchian channel or breaks the opposite band.
# This combination targets 15-30 trades per year per symbol (60-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1-day ATR(14) for volatility filter
    atr_len = 14
    if len(df_1d) < atr_len:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr).ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Donchian channel (20 periods) on 12h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    dc_mid = (dc_upper + dc_lower) / 2
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * price (avoid extremely low volatility)
        vol_filter = atr_1d_aligned[i] > 0.005 * close[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + volatility + volume
            if (close[i] > dc_upper[i] and 
                vol_filter and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + volatility + volume
            elif (close[i] < dc_lower[i] and 
                  vol_filter and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint or breaks below lower band
            if close[i] < dc_mid[i] or close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to midpoint or breaks above upper band
            if close[i] > dc_mid[i] or close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_ATR_Volume_Donchian_Breakout_v1"
timeframe = "12h"
leverage = 1.0