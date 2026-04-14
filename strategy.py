#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour ATR-based volatility filter.
# In low volatility (ATR ratio < 0.8), breakouts are more likely to fail; in high volatility (ATR ratio > 1.2),
# breakouts have stronger follow-through. We use the 12-hour ATR ratio (current ATR / 20-period average ATR)
# to filter entries, ensuring we only trade when volatility is expanding. This reduces false breakouts
# during ranging periods and improves performance in both bull and bear markets.
# Exit when price returns to the middle of the Donchian channel or breaks the opposite band.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for ATR-based volatility filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h ATR(14) for volatility measurement
    atr_len = 14
    if len(df_12h) < atr_len:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0  # first period has no prior close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    atr_12h_avg = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_12h / atr_12h_avg  # current ATR relative to average
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    
    # Donchian channel (20 periods) on 4h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    dc_middle = (dc_upper + dc_lower) / 2  # middle of the channel
    
    # Volume confirmation: current volume > 1.3x 20-period average (slightly lower threshold for more signals)
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
            np.isnan(atr_ratio_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: trade only when volatility is expanding (ATR ratio > 1.0)
        volatility_expanding = atr_ratio_aligned[i] > 1.0
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + volatility expanding + volume
            if (close[i] > dc_upper[i] and 
                volatility_expanding and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + volatility expanding + volume
            elif (close[i] < dc_lower[i] and 
                  volatility_expanding and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle of Donchian or breaks below lower band
            if close[i] < dc_middle[i] or close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle of Donchian or breaks above upper band
            if close[i] > dc_middle[i] or close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_ATR_Volatility_Donchian_Volume_v1"
timeframe = "4h"
leverage = 1.0