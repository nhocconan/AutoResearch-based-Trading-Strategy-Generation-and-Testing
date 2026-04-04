#!/usr/bin/env python3
"""
Experiment #3788: 12h Donchian(20) breakout + 1w EMA trend filter + 1d volume spike
HYPOTHESIS: 12h Donchian breakouts capture medium-term swings. 1w EMA(50) filters trend direction (only long in weekly uptrend, short in downtrend). 1d volume spike (>2.0x) confirms institutional participation. Works in bull markets (breakouts above resistance in uptrend) and bear markets (breakdowns below support in downtrend). Discrete position sizing (0.25) minimizes fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3788_12h_donchian20_1w_ema_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA(50) trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate weekly EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === HTF: 1d data for volume profile (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    # Calculate 1d volume MA(20)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 12h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 12h Indicators: Volume ratio for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                # Simplified: exit if price breaks below Donchian lower band
                if price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises above Donchian upper band (trend reversal)
                if price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume spike (> 2.0x 12h average) AND 1d volume above its 20-period MA
        volume_spike_12h = vol_ratio[i] > 2.0
        volume_confirm_1d = volume[i] > vol_ma_1d_aligned[i]
        
        if volume_spike_12h and volume_confirm_1d:
            # Long entry: Price breaks above Donchian upper band AND weekly EMA(50) is rising (uptrend)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]):  # Weekly EMA rising (uptrend)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND weekly EMA(50) is falling (downtrend)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]):  # Weekly EMA falling (downtrend)
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals