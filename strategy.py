#!/usr/bin/env python3
"""
1h_4h_Donchian_Breakout_HTFTrend_VolumeFilter
Hypothesis: 1h Donchian(20) breakout filtered by 4h EMA50 trend and volume spike.
Enter long when price breaks above 20-period high with 4h uptrend and above-average volume.
Enter short when price breaks below 20-period low with 4h downtrend and above-average volume.
Exit on opposite Donchian level break or ATR(14) trailing stop (2.0*ATR).
Uses 4h for signal direction, 1h only for entry timing and execution.
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
Works in bull/bear via 4h trend alignment and volume confirmation as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h EMA50 for HTF trend filter ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels: 20-period high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(donchian_high[i]) 
            or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current volume > 20-period average
            vol_confirm = volume[i] > vol_ma[i]
            
            # Long conditions: price > Donchian high, 4h uptrend, volume spike
            long_breakout = price > donchian_high[i]
            long_trend = price > ema_50_4h_aligned[i]
            
            # Short conditions: price < Donchian low, 4h downtrend, volume spike
            short_breakout = price < donchian_low[i]
            short_trend = price < ema_50_4h_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below Donchian low (support broken)
            elif price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above Donchian high (resistance broken)
            elif price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_Donchian_Breakout_HTFTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0