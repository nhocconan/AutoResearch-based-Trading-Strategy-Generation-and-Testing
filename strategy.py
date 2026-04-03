#!/usr/bin/env python3
"""
Experiment #292: 12h Donchian(20) breakout + 1d EMA(50) trend + 1w volume confirmation + ATR stoploss

HYPOTHESIS: Combining 12h Donchian breakouts with 1d EMA trend alignment and 1w volume confirmation creates a robust trend-following strategy that works in both bull and bear markets. The 12h timeframe minimizes fee drag while capturing medium-term trends, the 1d EMA provides trend filter to avoid counter-trend breakouts, and 1w volume confirmation ensures institutional participation. Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_ema_volume_v1"
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
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current volume / 20-period average volume) on 1w data
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.full(len(vol_1w), np.nan)
        valid = vol_ma_20_1w > 0
        vol_ratio_1w[valid] = vol_1w[valid] / vol_ma_20_1w[valid]
        
        # Align to 12h timeframe
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Calculate Donchian Channels (20-period)
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    
    for i in range(20-1, n):
        highest_high_20[i] = np.max(high[i-20+1:i+1])
        lowest_low_20[i] = np.min(low[i-20+1:i+1])
    
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
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
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
        # Volume confirmation: require above-average volume on 1w
        volume_confirmed = vol_ratio_1w_aligned[i] > 1.5
        
        # Long: Price breaks above Donchian upper band with 1d EMA uptrend and volume confirmation
        if close[i] > highest_high_20[i] and close[i] > ema_50_1d_aligned[i] and volume_confirmed:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Price breaks below Donchian lower band with 1d EMA downtrend and volume confirmation
        elif close[i] < lowest_low_20[i] and close[i] < ema_50_1d_aligned[i] and volume_confirmed:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals