#!/usr/bin/env python3
"""
Experiment #325: 12h Donchian(20) Breakout + 1d Volume Confirmation + ATR Trailing Stop

HYPOTHESIS: Price breaking above/below the 20-period Donchian channel on 12h timeframe indicates strong momentum.
Combined with 1d volume confirmation (>1.5x average) to ensure institutional participation, and filtered by 1d trend
(price > EMA50 for longs, < EMA50 for shorts). Uses ATR-based trailing stoploss (2.5*ATR) and exits at opposite
Donchian channel. Designed for low trade frequency (target: 12-37 trades/year on 12h) to minimize fee drag while
capturing significant trend moves in both bull and bear markets. The 1d timeframe HTF provides regime filter and
volume context, reducing false breakouts during choppy periods.
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
    
    # === HTF: 1d data for volume confirmation and trend filter (Call ONCE before loop) ===
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
    
    # Calculate EMA(50) on 1d close for trend filter
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    # Calculate highest high and lowest low over past 20 periods on 12h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_20
    donchian_lower = lowest_20
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0  # For trailing stop
    lowest_since_entry = 0.0   # For trailing stop
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 1d EMA50 ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Exit Logic (ATR-based trailing stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update highest price since entry
                highest_since_entry = max(highest_since_entry, high[i])
                # Trailing stop: highest price minus 2.5*ATR
                stop_level = highest_since_entry - 2.5 * atr_14
                # Exit if price hits stop or breaks below Donchian lower (contrarian exit)
                if low[i] < stop_level or close[i] < donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    highest_since_entry = 0.0
                    continue
            else:  # Short position
                # Update lowest price since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Trailing stop: lowest price plus 2.5*ATR
                stop_level = lowest_since_entry + 2.5 * atr_14
                # Exit if price hits stop or breaks above Donchian upper (contrarian exit)
                if high[i] > stop_level or close[i] > donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    lowest_since_entry = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper with volume confirmation and uptrend filter
        long_condition = (
            close[i] > donchian_upper[i] and  # Breakout above upper channel
            volume_spike and                  # Volume confirmation
            price_above_1d_ema                # Trend filter: above 1d EMA50
        )
        
        # Short: Price breaks below Donchian lower with volume confirmation and downtrend filter
        short_condition = (
            close[i] < donchian_lower[i] and  # Breakdown below lower channel
            volume_spike and                  # Volume confirmation
            price_below_1d_ema                # Trend filter: below 1d EMA50
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