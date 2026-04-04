#!/usr/bin/env python3
"""
Experiment #4116: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume spike
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts aligned with 1d EMA50 trend and confirmed by volume spikes capture sustained moves in both bull and bear markets. The 1d EMA50 acts as dynamic support/resistance, filtering counter-trend breakouts. Volume spike ensures breakout validity. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4116_12h_donchian20_1d_ema50_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    else:
        ema50_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel(20) for breakout levels ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 12h Indicators: Volume MA(20) for confirmation ===
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
    
    warmup = max(lookback_dc + 1, 50 + 10, 20 + 10)  # DC lookback, EMA buffer, vol MA buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                # Calculate ATR(20) on-the-fly for exit
                tr1 = high[max(0, i-19):i+1] - low[max(0, i-19):i+1]
                tr2 = np.abs(high[max(0, i-19):i+1] - close[max(0, i-19):i])
                tr3 = np.abs(low[max(0, i-19):i+1] - close[max(0, i-19):i])
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr_val = np.mean(tr) if len(tr) > 0 else 0.0
                if price < highest_since_entry - 2.5 * atr_val:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                tr1 = high[max(0, i-19):i+1] - low[max(0, i-19):i+1]
                tr2 = np.abs(high[max(0, i-19):i+1] - close[max(0, i-19):i])
                tr3 = np.abs(low[max(0, i-19):i+1] - close[max(0, i-19):i])
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr_val = np.mean(tr) if len(tr) > 0 else 0.0
                if price > lowest_since_entry + 2.5 * atr_val:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) to filter noise
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Breakout levels (use previous bar's levels to avoid look-ahead)
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # 1d trend filter
            above_1d_ema = price > ema50_1d_aligned[i]
            below_1d_ema = price < ema50_1d_aligned[i]
            
            # Long: breakout above Donchian high + above 1d EMA50
            long_entry = breakout_up and above_1d_ema
            # Short: breakout below Donchian low + below 1d EMA50
            short_entry = breakout_down and below_1d_ema
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
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
</trading_assistant>