#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_HTFTrend_VolumeSpike
Hypothesis: 12h Camarilla pivot (R1/S1) breakouts filtered by 1d EMA50 trend and volume spike.
Enter long when price breaks above 12h R1 with daily uptrend and above-average volume.
Enter short when price breaks below 12h S1 with daily downtrend and above-average volume.
Exit on ATR(14) trailing stop (2.5*ATR).
Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag.
Works in bull/bear via daily trend alignment and volume confirmation as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (daily for pivots/trend, 1h for volume MA)
    df_1d = get_htf_data(prices, '1d')
    df_1h = get_htf_data(prices, '1h')
    if len(df_1d) < 50 or len(df_1h) < 20:
        return np.zeros(n)
    
    # === 12h Camarilla Pivot Levels (R1, S1) ===
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_12h - low_12h) * 1.1 / 12.0
    r1_12h = close_12h + camarilla_range
    s1_12h = close_12h - camarilla_range
    
    # === Daily EMA50 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume spike filter (20-period 1h average) ===
    volume_1h = df_1h['volume'].values
    vol_ma_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_ma_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_1h)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1h_aligned[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current 12h volume > 20-period 1h volume MA (aligned)
            vol_confirm = prices['volume'].iloc[i] > vol_ma_1h_aligned[i]
            
            # Long conditions: price > 12h R1, daily uptrend, volume spike
            long_breakout = price > r1_12h[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < 12h S1, daily downtrend, volume spike
            short_breakout = price < s1_12h[i]
            short_trend = price < ema_50_1d_aligned[i]
            
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
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_HTFTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0