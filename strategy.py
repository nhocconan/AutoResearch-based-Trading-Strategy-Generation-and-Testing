#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_1dTrend_VolumeFilter
Hypothesis: 12h Camarilla pivot (R1/S1) breakouts filtered by 1d EMA34 trend and volume spike.
Enter long when price breaks above 12h R1 with 1d uptrend and above-average volume.
Enter short when price breaks below 12h S1 with 1d downtrend and above-average volume.
Exit on opposite level break or ATR(14) trailing stop (2.0*ATR).
Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
Works in bull/bear via 1d trend alignment and volume confirmation as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for pivots and trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 12h Camarilla Pivot Levels (R1, S1) ===
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Calculate 12h Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range_12h = (high_12h - low_12h) * 1.1 / 12.0
    r1_12h = close_12h + camarilla_range_12h
    s1_12h = close_12h - camarilla_range_12h
    
    # Align 12h levels to 12h timeframe (use previous completed 12h bar)
    r1_12h_aligned = align_htf_to_ltf(prices, prices, r1_12h)  # Same timeframe, no shift needed
    s1_12h_aligned = align_htf_to_ltf(prices, prices, s1_12h)
    
    # === 1d EMA34 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume spike filter (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
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
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) 
            or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current volume > 20-period average
            vol_confirm = volume[i] > vol_ma[i]
            
            # Long conditions: price > 12h R1, 1d uptrend, volume spike
            long_breakout = price > r1_12h_aligned[i]
            long_trend = price > ema_34_1d_aligned[i]
            
            # Short conditions: price < 12h S1, 1d downtrend, volume spike
            short_breakout = price < s1_12h_aligned[i]
            short_trend = price < ema_34_1d_aligned[i]
            
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
            # Trailing exit: price closes below 12h S1 (support broken)
            elif price < s1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 12h R1 (resistance broken)
            elif price > r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0