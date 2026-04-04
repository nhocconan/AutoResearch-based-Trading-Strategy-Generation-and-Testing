#!/usr/bin/env python3
"""
Experiment #4187: 6h Donchian(20) breakout + 1d pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with daily pivot bias (price above/below pivot) 
and volume spikes (>2x average) capture sustained moves with lower frequency. 
Uses discrete sizing (0.25) and trailing stops (2.5*ATR). Targets 50-150 trades over 4 years.
Works in bull/bear via pivot-based trend filter that only allows breakouts in direction 
of daily bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4187_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Precompute HTF: 1d Pivot (standard) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        typical = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
        pivot = typical.values
        r1 = 2 * pivot - df_1d['low'].values
        s1 = 2 * pivot - df_1d['high'].values
        # For trend bias: price > pivot = bullish, price < pivot = bearish
        pivot_bias = pivot  # use pivot as trend filter level
        pivot_bias_aligned = align_htf_to_ltf(prices, df_1d, pivot_bias)
    else:
        pivot_bias_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20 + 5, 20 + 5, 14 + 5)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(pivot_bias_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require strong volume confirmation (> 2x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Trend filter: price above daily pivot = bullish bias, price below = bearish bias
            bullish_trend = price > pivot_bias_aligned[i]
            bearish_trend = price < pivot_bias_aligned[i]
            
            # Long conditions: Donchian breakout up + bullish trend + volume confirmation
            long_entry = breakout_up and bullish_trend
            
            # Short conditions: Donchian breakout down + bearish trend + volume confirmation
            short_entry = breakout_dn and bearish_trend
            
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