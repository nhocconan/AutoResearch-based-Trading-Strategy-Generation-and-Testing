#!/usr/bin/env python3
"""
Experiment #5779: 6h Donchian(20) breakout + 12h Williams %R(14) mean reversion + volume confirmation
HYPOTHESIS: On 6h timeframe, price breaking above/below 20-period Donchian channel with volume > 1.5x average triggers entry in direction of breakout only when 12h Williams %R shows oversold/overbought conditions (Williams %R < -80 for longs, > -20 for shorts), indicating high-probability mean reversion within the breakout move. This combines breakout momentum with overextension filters to avoid false breakouts. Designed for 6h timeframe to balance trade frequency (target: 50-150 trades over 4 years) and work in both bull/bear markets by requiring volume confirmation and momentum alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5779_6h_donchian20_12h_willr_vol_v1"
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
    
    # === HTF: 12h data for Williams %R filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 14:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
        willr_12h = ((highest_high - close_12h) / (highest_high - lowest_low)) * -100
        willr_12h = np.where((highest_high - lowest_low) == 0, -50, willr_12h)  # avoid div by zero
    else:
        willr_12h = np.full(len(df_12h), -50.0)  # neutral
    
    # Align 12h Williams %R to 6h timeframe (shifted by 1 for completed 12h bars only)
    willr_12h_aligned = align_htf_to_ltf(prices, df_12h, willr_12h)
    
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
    
    warmup = max(20, 20, 14, 14)  # Donchian, volume avg, ATR, Williams %R period
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(willr_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low OR Williams %R becomes overbought (> -20)
                if price <= stop_price or price <= donchian_low[i] or willr_12h_aligned[i] > -20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high OR Williams %R becomes oversold (< -80)
                if price >= stop_price or price >= donchian_high[i] or willr_12h_aligned[i] < -80:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        willr_oversold = willr_12h_aligned[i] < -80  # Oversold condition
        willr_overbought = willr_12h_aligned[i] > -20  # Overbought condition
        
        # Entry conditions: breakout with volume confirmation AND Williams %R showing overextension in opposite direction
        # For longs: breakout up + volume + Williams %R oversold (price likely to mean revert up)
        # For shorts: breakout down + volume + Williams %R overbought (price likely to mean revert down)
        long_setup = breakout_up and volume_confirmed and willr_oversold
        short_setup = breakout_down and volume_confirmed and willr_overbought
        
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