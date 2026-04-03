#!/usr/bin/env python3
"""
Experiment #135: 6h Elder Ray Power + 1w Trend + Volume Filter

HYPOTHESIS: Elder Ray Bull Power (high - EMA13) and Bear Power (low - EMA13) on 6h timeframe,
combined with 1week trend filter (price > EMA50 for long, < EMA50 for short) and volume confirmation,
creates a robust strategy that captures institutional buying/selling pressure. Elder Ray measures
the power of bulls/bears behind each bar, working in both trending and ranging markets. The 1w trend
filter ensures we only take trades aligned with the higher timeframe direction, reducing false signals.
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume average (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period volume average on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    else:
        vol_ma_20_aligned = np.full(n, 1.0)
    
    # === 6h Indicators ===
    # Calculate EMA(13) for Elder Ray
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    else:
        ema_13 = np.full(n, np.nan)
    
    # Elder Ray Bull Power = High - EMA13
    bull_power = high - ema_13
    # Elder Ray Bear Power = Low - EMA13 (negative values indicate bear strength)
    bear_power = low - ema_13
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1w EMA50 ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume above 1d average ---
        volume_confirm = volume[i] > vol_ma_20_aligned[i]
        
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
                # Exit when Bull Power turns negative (bulls losing control)
                if bull_power[i] < 0:
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
                # Exit when Bear Power turns positive (bears losing control)
                if bear_power[i] > 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Bull Power positive (bulls in control) + uptrend + volume
        long_condition = (
            bull_power[i] > 0 and 
            price_above_1w_ema and 
            volume_confirm
        )
        
        # Short: Bear Power negative (bears in control) + downtrend + volume
        short_condition = (
            bear_power[i] < 0 and 
            price_below_1w_ema and 
            volume_confirm
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals