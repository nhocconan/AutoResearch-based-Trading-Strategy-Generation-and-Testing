#!/usr/bin/env python3
"""
Experiment #3808: 12h Donchian(20) breakout + 1w trend filter + volume confirmation
HYPOTHESIS: 12h Donchian breakouts capture medium-term swings. 1w EMA(50) trend filter ensures trades align with higher timeframe momentum. Volume confirmation (>1.5x average) filters low-participation breakouts. Works in bull markets (breakouts above resistance in uptrend) and bear markets (breakdowns below support in downtrend). Discrete position sizing (0.25) minimizes fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3808_12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w EMA(50) for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 12h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
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
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                # Calculate ATR(14) for exit condition
                if i >= 14:
                    prev_close = close[i-1]
                    tr = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
                    atr_14 = np.full(n, np.nan)
                    atr_14[14] = pd.Series([tr] + [0]*13).rolling(window=14, min_periods=14).mean().iloc[-1]
                    if i > 14:
                        atr_14[i] = (atr_14[i-1] * 13 + tr) / 14
                    else:
                        atr_14[i] = tr
                else:
                    atr_14_i = 0.0
                
                if price < highest_since_entry - 2.0 * atr_14_i:
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
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if i >= 14:
                    prev_close = close[i-1]
                    tr = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
                    atr_14 = np.full(n, np.nan)
                    atr_14[14] = pd.Series([tr] + [0]*13).rolling(window=14, min_periods=14).mean().iloc[-1]
                    if i > 14:
                        atr_14[i] = (atr_14[i-1] * 13 + tr) / 14
                    else:
                        atr_14[i] = tr
                else:
                    atr_14_i = 0.0
                
                if price > lowest_since_entry + 2.0 * atr_14_i:
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
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above 1w EMA(50) (bullish breakout in uptrend)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > ema_50_1w_aligned[i]): # Above 1w EMA(50) (uptrend filter)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below 1w EMA(50) (bearish breakdown in downtrend)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < ema_50_1w_aligned[i]): # Below 1w EMA(50) (downtrend filter)
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