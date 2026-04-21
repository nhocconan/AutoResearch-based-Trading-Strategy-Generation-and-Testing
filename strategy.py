#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike_v1
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot bias (price above/below weekly pivot) and volume confirmation (>2x 20-bar MA). 
In bull bias (price > weekly pivot), favor longs on upper band breakouts; in bear bias (price < weekly pivot), favor shorts on lower band breakdowns. 
ATR-based stoploss (2.0x) and discrete sizing (0.25) reduce churn. Target: 50-150 total trades over 4 years by requiring confluence of breakout, weekly bias, and volume.
Designed to work in bull (breakouts with bias) and bear (faded breakdowns vs bias) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1w Weekly Pivot (based on prior week OHLC) ===
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high[0] = prev_week_low[0] = prev_week_close[0] = np.nan  # first week invalid
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # === 6h Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        pivot_val = weekly_pivot_aligned[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        vol_conf = volume_confirmed[i]
        
        # Weekly bias
        is_bull_bias = price > pivot_val
        is_bear_bias = price < pivot_val
        
        if position == 0:
            if is_bull_bias:
                # Bull bias: long breakouts favored
                long_condition = (price > upper_val) and vol_conf
                short_condition = (price < lower_val) and vol_conf and (price < pivot_val * 0.995)  # stricter for shorts
            else:  # bear bias
                # Bear bias: short breakdowns favored
                short_condition = (price < lower_val) and vol_conf
                long_condition = (price > upper_val) and vol_conf and (price > pivot_val * 1.005)  # stricter for longs
            
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
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below lower band (failed breakout)
                elif price < lower_val:
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
                # Exit if price breaks above upper band (failed breakdown)
                elif price > upper_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0