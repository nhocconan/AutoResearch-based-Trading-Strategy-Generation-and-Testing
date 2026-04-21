#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_HTFTrend_ATRStop_V1
Hypothesis: 4h Donchian(20) breakout with 12h trend filter (price > 12h EMA34 for longs, < for shorts) and volume confirmation (>1.5x 20-period volume MA). 
ATR-based stoploss exits when price moves against position by 2.5x ATR(14). 
This combines price structure (Donchian), trend alignment (12h EMA), and volume confirmation to filter false breakouts. 
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag. 
Works in bull via breakouts and in bear via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA trend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # === 12h EMA34 for trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian(20) channels
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(34, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume confirmation + 12h uptrend
            if price > highest_20[i] and vol_ok and price > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower + volume confirmation + 12h downtrend
            elif price < lowest_20[i] and vol_ok and price < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position: trail stop or exit on trend reversal
            signals[i] = 0.25
            # Stoploss: price drops below entry - 2.5 * ATR
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if trend reverses
            elif price < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: trail stop or exit on trend reversal
            signals[i] = -0.25
            # Stoploss: price rises above entry + 2.5 * ATR
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if trend reverses
            elif price > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_VolumeSpike_HTFTrend_ATRStop_V1"
timeframe = "4h"
leverage = 1.0