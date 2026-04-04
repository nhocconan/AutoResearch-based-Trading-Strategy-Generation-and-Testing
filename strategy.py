#!/usr/bin/env python3
"""
Experiment #3899: 6h Donchian(20) breakout + 12h Supertrend(10,3) + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Supertrend direction capture medium-term momentum with reduced whipsaw. Volume > 1.5x MA(30) confirms participation. Works in bull/bear by following 12h Supertrend. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3899_6h_donchian20_12h_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Supertrend trend ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(10) for Supertrend
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Supertrend calculation
    hl2_12h = (high_12h + low_12h) / 2
    upper_12h = hl2_12h + (3 * atr_12h)
    lower_12h = hl2_12h - (3 * atr_12h)
    
    supertrend_12h = np.full_like(close_12h, np.nan, dtype=np.float64)
    direction_12h = np.ones_like(close_12h, dtype=np.int8)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(supertrend_12h[i-1]):
            supertrend_12h[i] = lower_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h[i] > supertrend_12h[i-1]:
                direction_12h[i] = 1
            else:
                direction_12h[i] = -1
            
            if direction_12h[i] == 1:
                supertrend_12h[i] = max(lower_12h[i], supertrend_12h[i-1])
            else:
                supertrend_12h[i] = min(upper_12h[i], supertrend_12h[i-1])
    
    # Align Supertrend direction to 6h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, direction_12h.astype(np.float64))
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(30) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[30:] = volume[30:] / vol_ma[30:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 30, 10)  # Donchian, vol MA, Supertrend ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(supertrend_dir_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
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
                if price > lowest_since_entry + 2.0 * atr[i]:
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
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine trend from 12h Supertrend: bullish if direction=1, bearish if direction=-1
            bullish = supertrend_dir_aligned[i] > 0
            bearish = supertrend_dir_aligned[i] < 0
            
            # Long entry: breakout above Donchian upper band in bullish regime
            long_breakout = price > highest_high[i-1] and bullish
            # Short entry: breakdown below Donchian lower band in bearish regime
            short_breakout = price < lowest_low[i-1] and bearish
            
            if long_breakout and not short_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_breakout and not long_breakout:
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

</think>