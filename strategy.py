#!/usr/bin/env python3
"""
Experiment #835: 6h Donchian(20) breakout + weekly pivot direction + volume spike
HYPOTHESIS: Donchian breakouts capture momentum; filtered by weekly pivot direction (bullish/bearish bias from prior week's high/low) and volume confirmation (>2.0x average). 
Long when price breaks above Donchian upper AND weekly pivot bullish AND volume spike. 
Short when price breaks below Donchian lower AND weekly pivot bearish AND volume spike. 
Weekly pivot provides structural bias: bullish if weekly close > weekly open, bearish if weekly close < weekly open. 
Uses discrete position sizing (0.25). Target: 75-200 total trades over 4 years (19-50/year).
Works in bull/bear markets: in bull trends, weekly pivot bias filters for longs; in bear trends, bias filters for shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_835_6h_donchian20_weeklypivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot bias (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot bias: 1 = bullish (close > open), -1 = bearish (close < open), 0 = neutral
    weekly_bias = np.zeros_like(close_1w)
    weekly_bias[1:] = np.where(close_1w[1:] > open_1w[1:], 1, 
                               np.where(close_1w[1:] < open_1w[1:], -1, 0))
    # Align bias to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20)  # sufficient for Donchian, volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(weekly_bias_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 4 bars (~24h on 6h) to avoid overtrading
            if bars_since_entry > 4:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long: price breaks above Donchian upper AND weekly pivot bullish
            if price > upper_20[i] and weekly_bias_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND weekly pivot bearish
            elif price < lower_20[i] and weekly_bias_aligned[i] < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals