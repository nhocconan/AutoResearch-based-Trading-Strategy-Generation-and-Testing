#!/usr/bin/env python3
"""
Experiment #1865: 12h Donchian(20) Breakout + Volume Spike + Chop Filter + ATR Stop
HYPOTHESIS: Donchian breakouts capture strong momentum moves. Combined with 1d trend filter (price > EMA50), volume confirmation (>2x average), and choppiness regime (CHOP < 61.8 = trending), this strategy avoids false breakouts in sideways markets. Discrete position sizing of 0.25 manages drawdown. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1865_12h_donchian20_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 12h Indicators: Donchian Channel(20) ===
    # Upper band: highest high of last 20 bars
    # Lower band: lowest low of last 20 bars
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: Choppiness Index(14) for regime filter ===
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula
    chop = np.zeros(n)
    for i in range(14, n):
        if highest_14[i] > lowest_14[i] and tr_sum_14[i] > 0:
            chop[i] = 100 * np.log10(tr_sum_14[i] / (highest_14[i] - lowest_14[i])) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral value when undefined
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop[i]) or
            np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss or opposite signal ---
        if in_position:
            bars_since_entry += 1
            
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Calculate ATR(14) for dynamic stoploss
            tr_atr = np.zeros(i+1)
            for j in range(1, i+1):
                tr_atr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_atr[0] = high[0] - low[0]
            atr_14 = pd.Series(tr_atr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5 * ATR below entry OR price breaks lowest_20
                if price <= entry_price - 2.5 * atr_14:
                    exit_signal = True
                elif price <= lowest_20[i]:  # Donchian lower band break
                    exit_signal = True
                # Optional: take profit at 3R
                elif price >= entry_price + 3.0 * 2.5 * atr_14:
                    exit_signal = True  # take profit and reverse if signal suggests
            else:  # Short position
                # Stoploss: 2.5 * ATR above entry OR price breaks highest_20
                if price >= entry_price + 2.5 * atr_14:
                    exit_signal = True
                elif price >= highest_20[i]:  # Donchian upper band break
                    exit_signal = True
                # Optional: take profit at 3R
                elif price <= entry_price - 3.0 * 2.5 * atr_14:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Require trending market (CHOP < 61.8 = trending, > 61.8 = choppy)
        trending = chop[i] < 61.8
        
        # Volume confirmation: require volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions
        breakout_up = price > highest_20[i-1]  # price breaks above upper band (using previous bar's value)
        breakout_down = price < lowest_20[i-1]  # price breaks below lower band
        
        if trending and volume_spike:
            if trend_bias > 0 and breakout_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif trend_bias < 0 and breakout_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals