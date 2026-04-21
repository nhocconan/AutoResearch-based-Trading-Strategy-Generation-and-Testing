#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike_v2
Hypothesis: 6h Donchian(20) breakouts in direction of weekly trend (price > weekly EMA34) with volume confirmation (>2.0x 20-bar MA). 
In bull weekly regime (price > weekly EMA34), take longs on upper Donchian breakouts; in bear weekly regime (price < weekly EMA34), take shorts on lower Donchian breakouts. 
Weekly EMA34 provides stable trend filter; Donchian breakouts capture momentum; volume filter ensures participation. 
Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year) by requiring confluence of breakout, weekly trend, and volume.
ATR-based stoploss (2.5x) and time-based exit (max 12 bars) control risk. Discrete sizing (0.25) minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for weekly trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 2.0x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma_20)
    
    # === 6h Donchian channels (20-period) based on PREVIOUS bar's high/low ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = prev_low[0] = np.nan  # first bar invalid
    
    upper_channel = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 12  # max 3 days (12 * 6h = 72h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        vol_conf = volume_confirmed[i]
        
        # Weekly trend regime
        is_bull = price > ema_34_1w_val
        is_bear = price < ema_34_1w_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long breakouts favored
                long_condition = (price > upper_val) and vol_conf
                short_condition = (price < lower_val) and vol_conf and (price < ema_34_1w_val * 0.99)  # stricter for shorts
            else:  # bear regime
                # Bear regime: short breakdowns favored
                short_condition = (price < lower_val) and vol_conf
                long_condition = (price > upper_val) and vol_conf and (price > ema_34_1w_val * 1.01)  # stricter for longs
            
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
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0