#!/usr/bin/env python3
"""
Experiment #5319: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, price breaking above/below the 20-period Donchian channel 
with volume > 1.8x average and aligned with the weekly pivot direction (from 1w timeframe) 
captures strong momentum moves while avoiding counter-trend whipsaws. Weekly pivot adds 
structural bias: long only when price above weekly pivot, short only when below. 
Uses discrete position sizing (0.25) and ATR-based trailing stop to control drawdown. 
Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5319_6h_donchian20_weekly_pivot_vol_v1"
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
    
    # === HTF: 1w data for weekly pivot direction ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 2:
        # Calculate weekly pivot from previous week's OHLC
        prev_close = df_1w['close'].shift(1).values
        prev_high = df_1w['high'].shift(1).values
        prev_low = df_1w['low'].shift(1).values
        
        # Weekly pivot point (standard calculation)
        weekly_pivot = (prev_high + prev_low + prev_close) / 3.0
        
        # Align to LTF (6h) with shift(1) for completed bars only
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 2)  # Donchian, volume avg, ATR, weekly pivot warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21-23 UTC) ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on trailing stoploss or failed breakout ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Trailing stop: 2.5 * ATR below highest since entry
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Trailing stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price crosses back below weekly pivot (trend change)
                if price <= stop_price or price <= donchian_low[i] or price < weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Trailing stop: 2.5 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Trailing stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price crosses back above weekly pivot (trend change)
                if price >= stop_price or price >= donchian_high[i] or price > weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions (using previous bar's levels)
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 1.8x average volume (tighter than 1.5x)
        volume_confirmed = volume_ratio[i] > 1.8
        
        # Weekly pivot filter: only long above weekly pivot, only short below
        pivot_long_filter = price > weekly_pivot_aligned[i-1]
        pivot_short_filter = price < weekly_pivot_aligned[i-1]
        
        # Entry conditions: Donchian breakout + volume + weekly pivot alignment
        camarilla_long = breakout_up and volume_confirmed and pivot_long_filter
        camarilla_short = breakout_down and volume_confirmed and pivot_short_filter
        
        if camarilla_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif camarilla_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #5319: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, price breaking above/below the 20-period Donchian channel 
with volume > 1.8x average and aligned with the weekly pivot direction (from 1w timeframe) 
captures strong momentum moves while avoiding counter-trend whipsaws. Weekly pivot adds 
structural bias: long only when price above weekly pivot, short only when below. 
Uses discrete position sizing (0.25) and ATR-based trailing stop to control drawdown. 
Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5319_6h_donchian20_weekly_pivot_vol_v1"
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
    
    # === HTF: 1w data for weekly pivot direction ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 2:
        # Calculate weekly pivot from previous week's OHLC
        prev_close = df_1w['close'].shift(1).values
        prev_high = df_1w['high'].shift(1).values
        prev_low = df_1w['low'].shift(1).values
        
        # Weekly pivot point (standard calculation)
        weekly_pivot = (prev_high + prev_low + prev_close) / 3.0
        
        # Align to LTF (6h) with shift(1) for completed bars only
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 2)  # Donchian, volume avg, ATR, weekly pivot warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21-23 UTC) ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on trailing stoploss or failed breakout ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Trailing stop: 2.5 * ATR below highest since entry
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Trailing stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price crosses back below weekly pivot (trend change)
                if price <= stop_price or price <= donchian_low[i] or price < weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Trailing stop: 2.5 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Trailing stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price crosses back above weekly pivot (trend change)
                if price >= stop_price or price >= donchian_high[i] or price > weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions (using previous bar's levels)
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 1.8x average volume (tighter than 1.5x)
        volume_confirmed = volume_ratio[i] > 1.8
        
        # Weekly pivot filter: only long above weekly pivot, only short below
        pivot_long_filter = price > weekly_pivot_aligned[i-1]
        pivot_short_filter = price < weekly_pivot_aligned[i-1]
        
        # Entry conditions: Donchian breakout + volume + weekly pivot alignment
        camarilla_long = breakout_up and volume_confirmed and pivot_long_filter
        camarilla_short = breakout_down and volume_confirmed and pivot_short_filter
        
        if camarilla_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif camarilla_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals