#!/usr/bin/env python3
"""
Experiment #5699: 6h Donchian(20) breakout + 12h Supertrend filter + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 2.0x average and aligned 
with 12h Supertrend trend direction capture high-probability continuation moves. 
Supertrend provides adaptive trend filtering that works in both bull and bear markets 
by adjusting to volatility. Volume confirms breakout strength. ATR trailing stop (2.5x) 
manages risk. Discrete sizing (0.25) minimizes fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5699_6h_donchian20_12h_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 12h data for Supertrend calculation ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 10:
        # Supertrend calculation on 12h data
        period = 10
        multiplier = 3.0
        
        hl2 = (df_12h['high'] + df_12h['low']) / 2
        tr1 = df_12h['high'] - df_12h['low']
        tr2 = np.abs(df_12h['high'] - np.roll(df_12h['close'], 1))
        tr3 = np.abs(df_12h['low'] - np.roll(df_12h['close'], 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr_12h = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        
        upperband = hl2 + (multiplier * atr_12h)
        lowerband = hl2 - (multiplier * atr_12h)
        
        supertrend = np.full(len(df_12h), np.nan)
        direction = np.full(len(df_12h), np.nan)  # 1 for uptrend, -1 for downtrend
        
        for i in range(period, len(df_12h)):
            # Upper band logic
            if upperband[i] < upperband[i-1] or df_12h['close'].iloc[i-1] > upperband[i-1]:
                upperband[i] = hl2[i] + (multiplier * atr_12h[i])
            
            # Lower band logic
            if lowerband[i] > lowerband[i-1] or df_12h['close'].iloc[i-1] < lowerband[i-1]:
                lowerband[i] = hl2[i] - (multiplier * atr_12h[i])
            
            # Supertrend logic
            if supertrend[i-1] == upperband[i-1]:
                if df_12h['close'].iloc[i] <= upperband[i]:
                    supertrend[i] = upperband[i]
                else:
                    supertrend[i] = lowerband[i]
                    direction[i] = 1
            else:
                if df_12h['close'].iloc[i] >= lowerband[i]:
                    supertrend[i] = lowerband[i]
                    direction[i] = 1
                else:
                    supertrend[i] = upperband[i]
                    direction[i] = -1
            
            # Initialize first value
            if np.isnan(direction[i]):
                direction[i] = 1 if df_12h['close'].iloc[i] > supertrend[i] else -1
        
        # For first period values, set direction based on close vs hl2
        if np.isnan(direction[period-1]):
            direction[period-1] = 1 if df_12h['close'].iloc[period-1] > hl2.iloc[period-1] else -1
            supertrend[period-1] = lowerband[period-1] if direction[period-1] == 1 else upperband[period-1]
        
        # Fill any remaining NaNs with previous direction
        for i in range(period, len(direction)):
            if np.isnan(direction[i]):
                direction[i] = direction[i-1]
    else:
        direction = np.full(len(df_12h), np.nan)
    
    # Align 12h Supertrend direction to 6h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 10)  # Donchian, volume avg, ATR, Supertrend lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(supertrend_direction_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR Supertrend turns bearish
                if price <= stop_price or supertrend_direction_aligned[i] <= 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR Supertrend turns bullish
                if price >= stop_price or supertrend_direction_aligned[i] >= 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Supertrend bias: long when uptrend (1), short when downtrend (-1)
        long_bias = supertrend_direction_aligned[i] > 0
        short_bias = supertrend_direction_aligned[i] < 0
        
        # Entry conditions: breakout in direction of Supertrend with volume
        long_setup = breakout_up and volume_confirmed and long_bias
        short_setup = breakout_down and volume_confirmed and short_bias
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals