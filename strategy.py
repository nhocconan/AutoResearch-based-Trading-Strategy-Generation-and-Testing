#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeFilter
Hypothesis: 6h Donchian(20) breakouts filtered by weekly Camarilla pivot trend and volume confirmation.
Enter long when price breaks above 6h Donchian upper band with weekly bullish bias (price above weekly R1) and volume spike.
Enter short when price breaks below 6h Donchian lower band with weekly bearish bias (price below weekly S1) and volume spike.
Exit on opposite band break or ATR(14) trailing stop (2.0*ATR).
Designed for low trade frequency (target: 12-30 trades/year) to minimize fee drag.
Works in bull/bear via weekly pivot alignment as regime filter and volume confirmation as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (daily for pivots, weekly for trend)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 6h Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian: upper = max(high, 20), lower = min(low, 20)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly Camarilla Pivot Levels (R1, S1) for trend filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range_1w = (high_1w - low_1w) * 1.1 / 12.0
    r1_1w = close_1w + camarilla_range_1w
    s1_1w = close_1w - camarilla_range_1w
    
    # Align weekly R1/S1 to 6h timeframe (use previous completed weekly bar)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === Volume spike filter (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) 
            or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current volume > 20-period average
            vol_confirm = volume[i] > vol_ma[i]
            
            # Long conditions: price > 6h Donchian upper, weekly bullish bias (above R1), volume spike
            long_breakout = price > high_roll[i]
            long_trend = price > r1_1w_aligned[i]
            
            # Short conditions: price < 6h Donchian lower, weekly bearish bias (below S1), volume spike
            short_breakout = price < low_roll[i]
            short_trend = price < s1_1w_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 6h Donchian lower (support broken)
            elif price < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 6h Donchian upper (resistance broken)
            elif price > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0