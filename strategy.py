#!/usr/bin/env python3
"""
Experiment #4179: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot bias 
(above/below weekly pivot) and confirmed by volume spikes (>1.8x average) provide 
high-probability trend continuation entries. Uses 0.25 position size to balance 
drawdown and returns. Weekly pivot acts as regime filter: price above weekly pivot 
= bullish bias (long breakouts), below = bearish bias (short breakouts). Volume 
confirmation filters false breakouts. Works in both bull/bear via pivot regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4179_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1w data for weekly pivot calculation ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Align to LTF: each 6h bar gets the pivot from the completed weekly bar
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period):
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
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
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
        # Require volume confirmation (> 1.8x average) to filter false breakouts
        volume_confirm = vol_ratio[i] > 1.8
        
        if volume_confirm:
            # Donchian breakout conditions
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Weekly pivot regime filter
            price_above_pivot = price > weekly_pivot_aligned[i]
            price_below_pivot = price < weekly_pivot_aligned[i]
            
            # Long: breakout above Donchian upper + price above weekly pivot (bullish regime)
            long_entry = breakout_up and price_above_pivot
            
            # Short: breakout below Donchian lower + price below weekly pivot (bearish regime)
            short_entry = breakout_dn and price_below_pivot
            
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