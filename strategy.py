#!/usr/bin/env python3
"""
Experiment #031: 6h ATR Breakout + Volume Spike + 1d Trend Filter

HYPOTHESIS: Combining ATR-based volatility breakouts with volume confirmation and 1d trend alignment 
creates a robust strategy for 6h timeframe. The ATR breakout captures significant price moves, 
volume spike confirms institutional participation, and the 1d trend filter ensures we only trade 
in the direction of the higher timeframe trend. This approach works in both bull and bear markets 
by adapting to volatility regimes and avoiding counter-trend whipsaws. Targets 12-37 trades/year 
on 6h timeframe (50-150 total over 4 years) to minimize fee drag while capturing high-momentum moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_atr_breakout_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Calculate ATR(14) for volatility breakout
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate ATR(14) moving average for volatility regime filter
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / atr_ma_20  # Current ATR vs 20-period average
    
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
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when volatility is elevated (> 1.3x average ATR) ---
        volatility_expansion = atr_ratio[i] > 1.3
        
        # --- Trend Filter: Align with 1d EMA50 direction ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike ---
        # Calculate volume ratio on 6h timeframe
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            vol_ratio = volume[i] / vol_ma_20 if vol_ma_20 > 0 else 1.0
        else:
            vol_ratio = 1.0
        volume_spike = vol_ratio > 1.5
        
        # --- Breakout Conditions ---
        # Upper breakout: close > previous close + ATR multiplier
        upper_breakout = close[i] > close[i-1] + 0.5 * atr_14[i]
        # Lower breakout: close < previous close - ATR multiplier
        lower_breakout = close[i] < close[i-1] - 0.5 * atr_14[i]
        
        # --- Exit Logic (ATR-based stoploss and time-based exit) ---
        if in_position:
            # Calculate current ATR for stoploss
            current_atr = atr_14[i]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * current_atr
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Time-based exit: max 3 bars (18 hours) to prevent overstaying
                if i - entry_bar >= 3:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * current_atr
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Time-based exit: max 3 bars (18 hours)
                if i - entry_bar >= 3:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Volatility expansion + volume spike + upward breakout + uptrend filter
        long_condition = (
            volatility_expansion and 
            volume_spike and 
            upper_breakout and 
            price_above_1d_ema
        )
        
        # Short: Volatility expansion + volume spike + downward breakout + downtrend filter
        short_condition = (
            volatility_expansion and 
            volume_spike and 
            lower_breakout and 
            price_below_1d_ema
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals