#!/usr/bin/env python3
"""
4h Donchian20 Breakout + Volume Spike + 1d Supertrend Filter
Hypothesis: Donchian channel breakouts capture strong momentum moves. Volume spike confirms institutional participation. 1d Supertrend filter ensures alignment with daily trend, reducing false breakouts in choppy or ranging markets. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 30-50 trades/year on 4h.
Works in bull markets via breakouts with trend and in bear markets via trend filter (avoids counter-trend entries).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Supertrend calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Supertrend (ATR=10, mult=3.0)
    hl2 = (df_1d['high'] + df_1d['low']) / 2
    atr = pd.Series(df_1d['high'] - df_1d['low']).rolling(window=10, min_periods=10).mean()
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    supertrend = np.zeros(len(df_1d))
    direction = np.ones(len(df_1d))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_1d)):
        if close_1d := df_1d['close'].iloc[i]:
            pass
        # Supertrend logic
        if df_1d['close'].iloc[i] > upperband.iloc[i-1]:
            direction[i] = 1
        elif df_1d['close'].iloc[i] < lowerband.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = lowerband.iloc[i-1]
            if direction[i] == -1 and upperband.iloc[i] > upperband.iloc[i-1]:
                upperband.iloc[i] = upperband.iloc[i-1]
    
        if direction[i] == 1:
            supertrend[i] = lowerband.iloc[i]
        else:
            supertrend[i] = upperband.iloc[i]
    
    # Align Supertrend to 4h
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    
    # Calculate Donchian channel (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper_donchian = donchian_high[i]
        lower_donchian = donchian_low[i]
        st_value = supertrend_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian AND volume spike AND price > Supertrend (uptrend)
            long_entry = (curr_close > upper_donchian) and vol_spike and (curr_close > st_value)
            # Short: price breaks below lower Donchian AND volume spike AND price < Supertrend (downtrend)
            short_entry = (curr_close < lower_donchian) and vol_spike and (curr_close < st_value)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price closes below lower Donchian (reversal) OR price < Supertrend (trend change)
            if (curr_close < lower_donchian) or (curr_close < st_value):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price closes above upper Donchian (reversal) OR price > Supertrend (trend change)
            if (curr_close > upper_donchian) or (curr_close > st_value):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_1dSupertrend_Trend"
timeframe = "4h"
leverage = 1.0