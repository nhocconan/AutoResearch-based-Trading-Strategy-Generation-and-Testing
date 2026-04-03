#!/usr/bin/env python3
"""
Experiment #252: 12h Donchian Breakout + 1d EMA Trend + 1w Choppiness Regime

HYPOTHESIS: Using 12h timeframe with Donchian(20) breakouts for entry, aligned with 1d EMA(50) trend filter and 1w Choppiness Index regime filter. This captures medium-term breakouts in trending markets while avoiding ranging conditions. The 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. Donchian breakouts provide clear entry/exit levels, EMA trend ensures directional bias, and weekly chop filter avoids false breakouts in sideways markets. Works in both bull (breakouts continuation) and bear (breakdowns continuation) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_ema_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for choppiness regime (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index(14) on 1w data
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr_1w = np.zeros(len(close_1w))
        tr_1w[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        
        # Sum of TR over 14 periods
        sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index = 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
        chop_1w = np.full(len(close_1w), np.nan)
        valid = (sum_tr_14 > 0) & (max_high_14 > min_low_14) & ~(np.isnan(sum_tr_14) | np.isnan(max_high_14) | np.isnan(min_low_14))
        chop_1w[valid] = 100 * np.log10(sum_tr_1w[valid] / (max_high_14[valid] - min_low_14[valid])) / np.log10(14)
        
        # Align to 12h timeframe
        chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    else:
        chop_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel(20) ===
    # Calculate Donchian Channel on 12h data directly
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Avoid choppy markets (Choppiness > 61.8 = ranging) ---
        # Only trade when market is trending (Choppiness < 38.2) or moderate (38.2-61.8)
        # Avoid strong ranging regimes where breakouts fail
        if chop_1w_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        # --- Price Trend Alignment ---
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Donchian Breakout Signals ---
        # Long breakout: price closes above upper Donchian band
        long_breakout = close[i] > highest_high[i]
        # Short breakout: price closes below lower Donchian band
        short_breakout = close[i] < lowest_low[i]
        
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
                # Take profit at 3R or when price hits opposite Donchian band
                if close[i] >= entry_price + 3.0 * 2.5 * atr_14 or close[i] < lowest_low[i]:
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
                # Take profit at 3R or when price hits opposite Donchian band
                if close[i] <= entry_price - 3.0 * 2.5 * atr_14 or close[i] > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above upper band with price above 1d EMA (bullish alignment)
        if long_breakout and price_above_ema:
            in_position = True
            position_side = 1
            entry_bar = i
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Donchian breakout below lower band with price below 1d EMA (bearish alignment)
        elif short_breakout and price_below_ema:
            in_position = True
            position_side = -1
            entry_bar = i
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals