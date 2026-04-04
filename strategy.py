#!/usr/bin/env python3
"""
Experiment #3835: 6h Donchian(20) breakout + 1w trend filter + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture medium-term swings with weekly trend alignment (price > weekly EMA20) and volume confirmation (>2.0x) to filter false breakouts. 
Weekly EMA20 ensures we only trade in the direction of the longer-term trend, reducing whipsaws in ranging markets. 
Volume spike confirms institutional participation. Discrete position sizing (0.25) minimizes fee drag. 
Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3835_6h_donchian20_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA to 6h timeframe (shifted by 1 for completed weekly bar)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                # Calculate ATR manually for exit condition
                if i > 0:
                    atr_temp = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                    if price < highest_since_entry - 2.0 * atr_temp:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price breaks below Donchian lower band (trend reversal)
                    elif price < lowest_low[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if i > 0:
                    atr_temp = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                    if price > lowest_since_entry + 2.0 * atr_temp:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price breaks above Donchian upper band (trend reversal)
                    elif price > highest_high[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) and weekly trend alignment
        volume_spike = vol_ratio[i] > 2.0
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND weekly uptrend
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                weekly_uptrend):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND weekly downtrend
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  weekly_downtrend):
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