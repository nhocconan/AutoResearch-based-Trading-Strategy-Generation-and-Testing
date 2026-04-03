#!/usr/bin/env python3
"""
Experiment #259: 6h Donchian(20) Breakout + 12h EMA Trend + 1d Volume Spike Filter

HYPOTHESIS: Combining 6h Donchian breakouts with 12h EMA trend alignment and 1d volume confirmation creates a robust trend-following strategy. The 6h Donchian(20) captures medium-term breakouts, the 12h EMA ensures alignment with the longer-term trend to avoid counter-trend trades, and the 1d volume spike filter confirms institutional participation. This approach works in both bull and bear markets by trading breakouts in the direction of the 12h trend, avoiding false breakouts during low-volume periods. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(50) on 12h close
    if len(df_12h) >= 50:
        close_12h = df_12h['close'].values
        ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    else:
        ema_50_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Volume Spike Ratio: volume / volume_ma(20)
    if len(df_1d) >= 20:
        volume_1d = df_1d['volume'].values
        volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike_ratio = np.full(len(volume_1d), np.nan)
        valid = volume_ma_20 > 0
        volume_spike_ratio[valid] = volume_1d[valid] / volume_ma_20[valid]
        
        # Align to 6h timeframe
        volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_ratio)
    else:
        volume_spike_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Donchian Channel(20)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Filter: Require volume spike > 1.5x average ---
        if volume_spike_aligned[i] < 1.5:
            signals[i] = 0.0
            continue
        
        # --- Price Trend Alignment (12h EMA) ---
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # --- Donchian Breakout Signals ---
        # Long breakout: price breaks above 20-period high
        long_breakout = close[i] > highest_high_20[i-1] if i > 0 else False
        # Short breakout: price breaks below 20-period low
        short_breakout = close[i] < lowest_low_20[i-1] if i > 0 else False
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3R (7.5 * ATR)
                if high[i] > entry_price + 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3R (7.5 * ATR)
                if low[i] < entry_price - 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above + price above 12h EMA (bullish alignment)
        if long_breakout and price_above_ema:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakout below + price below 12h EMA (bearish alignment)
        elif short_breakout and price_below_ema:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals