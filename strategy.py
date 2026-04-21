#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_EMA34_VolumeSpike_ATRStop
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation (>1.8x 20-period MA).
Long when price breaks above 20-period high, above 1d EMA34, and volume > 1.8x average.
Short when price breaks below 20-period low, below 1d EMA34, and volume > 1.8x average.
Uses ATR-based stop (2.0x) and minimum holding period of 3 bars to reduce churn.
Designed for 12h timeframe to target 50-150 trades over 4 years (12-37/year) with discrete sizing 0.25.
Combines price channel structure (Donchian) with trend filter (1d EMA34) and volume confirmation to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.8x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 12h Donchian channels (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        vol_avg = vol_ma[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        
        # Volume confirmation: current volume > 1.8x average (tighter threshold)
        volume_confirm = volume_now > 1.8 * vol_avg
        
        if position == 0:
            # Long: price breaks above 20-period high, above 1d EMA34, volume confirm
            long_condition = (price > highest_high_val) and (price > ema_34_1d_val) and volume_confirm
            # Short: price breaks below 20-period low, below 1d EMA34, volume confirm
            short_condition = (price < lowest_low_val) and (price < ema_34_1d_val) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below 1d EMA34)
                elif price < ema_34_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above 1d EMA34)
                elif price > ema_34_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_EMA34_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0