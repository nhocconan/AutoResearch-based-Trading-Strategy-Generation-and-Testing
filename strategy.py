#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_12hTrend_VolumeFilter
Hypothesis: 6h Camarilla pivot (R1/S1) breakouts filtered by 12h EMA34 trend and volume spike.
Enter long when price breaks above 6h R1 with 12h uptrend and above-average volume.
Enter short when price breaks below 6h S1 with 12h downtrend and above-average volume.
Exit on opposite level break or ATR(14) trailing stop (2.0*ATR).
Designed for low trade frequency (target: 12-25 trades/year) to minimize fee drag.
Uses 6h primary timeframe with 12h HTF trend filter and volume confirmation for regime.
Works in bull/bear via 12h trend alignment and volume confirmation as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (6h for pivots, 12h for trend)
    df_6h = get_htf_data(prices, '6h')
    df_12h = get_htf_data(prices, '12h')
    if len(df_6h) < 20 or len(df_12h) < 20:
        return np.zeros(n)
    
    # === 6h Camarilla Pivot Levels (R1, S1) ===
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_6h - low_6h) * 1.1 / 12.0
    r1_6h = close_6h + camarilla_range
    s1_6h = close_6h - camarilla_range
    
    # Align to 6h timeframe (use previous completed 6h bar)
    r1_6h_aligned = align_htf_to_ltf(prices, df_6h, r1_6h)
    s1_6h_aligned = align_htf_to_ltf(prices, df_6h, s1_6h)
    
    # === 12h EMA34 for HTF trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === Volume spike filter (20-period average on 6h) ===
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
        if (np.isnan(r1_6h_aligned[i]) or np.isnan(s1_6h_aligned[i]) 
            or np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current volume > 20-period average
            vol_confirm = volume[i] > vol_ma[i]
            
            # Long conditions: price > 6h R1, 12h uptrend, volume spike
            long_breakout = price > r1_6h_aligned[i]
            long_trend = price > ema_34_12h_aligned[i]
            
            # Short conditions: price < 6h S1, 12h downtrend, volume spike
            short_breakout = price < s1_6h_aligned[i]
            short_trend = price < ema_34_12h_aligned[i]
            
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
            # Trailing exit: price closes below 6h S1 (support broken)
            elif price < s1_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 6h R1 (resistance broken)
            elif price > r1_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0