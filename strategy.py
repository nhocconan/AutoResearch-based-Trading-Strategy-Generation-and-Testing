#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_v2
Hypothesis: On 4h timeframe, trade Donchian(20) breakouts with 1d EMA34 trend filter and volume spike confirmation. Daily trend filter provides medium-term bias to reduce false breakouts. Volume spike confirms institutional participation. ATR-based stoploss manages risk. Designed for 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.25) to minimize fee drag. Works in bull/bear markets via daily trend filter.
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
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(34) 1d, Donchian(20), ATR(14), volume MA (20)
    start_idx = max(34, 20, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        atr_val = atr[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
        if position == 0:
            # Long: break above Donchian high with uptrend and volume spike
            long_signal = (high_val > highest_high_val) and uptrend and vol_spike
            
            # Short: break below Donchian low with downtrend and volume spike
            short_signal = (low_val < lowest_low_val) and downtrend and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
                lowest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                highest_since_entry = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            lowest_since_entry = min(lowest_since_entry, low_val)
            # Exit: trend reversal or ATR stoploss
            if (close_val < ema_34_1d_val or 
                close_val <= entry_price - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            highest_since_entry = max(highest_since_entry, high_val)
            lowest_since_entry = min(lowest_since_entry, low_val)
            # Exit: trend reversal or ATR stoploss
            if (close_val > ema_34_1d_val or 
                close_val >= entry_price + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0