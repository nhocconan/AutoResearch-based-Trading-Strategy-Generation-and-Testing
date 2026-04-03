#!/usr/bin/env python3
"""
Experiment #376: 12h Donchian Breakout + 1d Volume Spike + ATR Trend Filter

HYPOTHESIS: Donchian(20) breakout on 12h timeframe, confirmed by 1d volume spike (>2x average) 
and filtered by 1d price > EMA50 for longs / < EMA50 for shorts, captures strong momentum 
moves in both bull and bear markets. The Donchian structure provides objective breakout 
levels, volume confirms institutional participation, and the EMA50 filter ensures alignment 
with higher timeframe trend. Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years) 
to minimize fee drag while capturing high-probability trends. Uses ATR-based trailing stoploss 
and discrete position sizing (0.30) to control drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike and EMA50 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20) ===
    # Calculate rolling max/min on 12h high/low
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0  # For trailing stop
    lowest_since_entry = 0.0   # For trailing stop
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 1d EMA50 ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (Trailing stoploss) ---
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Trail stop: exit if price drops 2.5*ATR from highest since entry
                # Calculate ATR(14) for stoploss
                tr = np.zeros(i+1)
                tr[0] = high[0] - low[0]
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
                stop_level = highest_since_entry - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Calculate ATR(14) for stoploss
                tr = np.zeros(i+1)
                tr[0] = high[0] - low[0]
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
                stop_level = lowest_since_entry + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian(20) high with volume spike and price > 1d EMA50
        long_condition = (
            close[i] > highest_20[i] and  # Breakout above Donchian high
            volume_spike and              # Volume confirmation
            price_above_1d_ema            # Trend filter: uptrend
        )
        
        # Short: Price breaks below Donchian(20) low with volume spike and price < 1d EMA50
        short_condition = (
            close[i] < lowest_20[i] and   # Breakdown below Donchian low
            volume_spike and              # Volume confirmation
            price_below_1d_ema            # Trend filter: downtrend
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
</file>