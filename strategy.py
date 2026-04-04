#!/usr/bin/env python3
"""
Experiment #4499: 6h Donchian(20) Breakout + 12h Supertrend + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with 12h Supertrend direction and confirmed by volume (>2.0x average) capture medium-term momentum with reduced whipsaws. Supertrend on 12h provides robust trend filtering that works in both bull and bear markets by adapting to volatility. Volume filters low-conviction moves. Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25. The 6h timeframe reduces noise while the 12h Supertrend ensures alignment with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4499_6h_donchian20_12h_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 12h data for Supertrend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 1:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # Calculate ATR(10) for Supertrend
        tr1_12h = high_12h[1:] - low_12h[1:]
        tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
        tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
        tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
        atr_12h = pd.Series(tr_12h).ewm(span=10, min_periods=10, adjust=False).mean().values
        
        # Supertrend parameters
        atr_mult = 3.0
        
        # Basic upper and lower bands
        hl2_12h = (high_12h + low_12h) / 2.0
        upper_band_12h = hl2_12h + (atr_mult * atr_12h)
        lower_band_12h = hl2_12h - (atr_mult * atr_12h)
        
        # Initialize Supertrend arrays
        supertrend_12h = np.full_like(close_12h, np.nan)
        direction_12h = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
        
        # Calculate Supertrend
        for i in range(1, len(close_12h)):
            if np.isnan(atr_12h[i]) or np.isnan(upper_band_12h[i]) or np.isnan(lower_band_12h[i]):
                continue
                
            # Upper band logic
            if close_12h[i-1] <= upper_band_12h[i-1]:
                upper_band_12h[i] = min(upper_band_12h[i], upper_band_12h[i-1])
            else:
                upper_band_12h[i] = upper_band_12h[i]
                
            # Lower band logic
            if close_12h[i-1] >= lower_band_12h[i-1]:
                lower_band_12h[i] = max(lower_band_12h[i], lower_band_12h[i-1])
            else:
                lower_band_12h[i] = lower_band_12h[i]
                
            # Supertrend logic
            if i == 1:
                # Initialize first value
                if close_12h[i] > upper_band_12h[i]:
                    direction_12h[i] = -1  # downtrend
                    supertrend_12h[i] = upper_band_12h[i]
                else:
                    direction_12h[i] = 1   # uptrend
                    supertrend_12h[i] = lower_band_12h[i]
            else:
                if direction_12h[i-1] == 1:  # previous uptrend
                    if close_12h[i] <= supertrend_12h[i-1]:
                        direction_12h[i] = -1  # change to downtrend
                        supertrend_12h[i] = upper_band_12h[i]
                    else:
                        direction_12h[i] = 1   # remain uptrend
                        supertrend_12h[i] = max(lower_band_12h[i], supertrend_12h[i-1])
                else:  # previous downtrend
                    if close_12h[i] >= supertrend_12h[i-1]:
                        direction_12h[i] = 1   # change to uptrend
                        supertrend_12h[i] = lower_band_12h[i]
                    else:
                        direction_12h[i] = -1  # remain downtrend
                        supertrend_12h[i] = min(upper_band_12h[i], supertrend_12h[i-1])
        
        # Align Supertrend direction to 6h timeframe
        supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    else:
        supertrend_dir_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 10)  # Donchian, vol MA, ATR, Supertrend init
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(supertrend_dir_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        # Supertrend bias: 1 = uptrend (long bias), -1 = downtrend (short bias)
        long_bias = supertrend_dir_aligned[i] == 1
        short_bias = supertrend_dir_aligned[i] == -1
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Long conditions: upward breakout + long bias + volume
        long_entry = breakout_up and long_bias and volume_confirm
        
        # Short conditions: downward breakout + short bias + volume
        short_entry = breakout_down and short_bias and volume_confirm
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals