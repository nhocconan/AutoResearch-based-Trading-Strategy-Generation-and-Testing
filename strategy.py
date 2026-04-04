#!/usr/bin/env python3
"""
Experiment #6231: 6h Donchian(20) breakout + 1d Weekly Pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R1/S1) capture institutional 
flow while avoiding false breakouts. Weekly pivots provide key support/resistance from larger 
participants. Volume >1.5x average confirms participation. Designed for 6h timeframe to target 
50-150 trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
Works in both bull/bear by using pivot direction as trend filter and Donchian for breakout timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6231_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # === HTF: 1d data for Weekly Pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot points from prior week's OHLC
        # Need at least 5 days (1 week) of data
        weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
        weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().shift(1)
        weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).mean().shift(1)
        
        # Weekly Pivot Point (PP) = (H + L + C) / 3
        pp = (weekly_high + weekly_low + weekly_close) / 3.0
        # R1 = (2 * PP) - L, S1 = (2 * PP) - H
        r1 = (2 * pp) - weekly_low
        s1 = (2 * pp) - weekly_high
        
        # Align to 6h timeframe
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    else:
        pp_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 5) + 1  # Donchian, volume avg, ATR, weekly lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21:00-23:59 UTC) ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5  # Volume filter for stronger signals
        
        # Weekly Pivot direction filter: price relative to PP
        bullish_bias = price > pp_aligned[i]
        bearish_bias = price < pp_aligned[i]
        
        # Entry conditions: breakout with volume AND pivot alignment
        # Long: breakout up with volume AND price above PP (bullish bias)
        # Short: breakout down with volume AND price below PP (bearish bias)
        long_entry = breakout_up and volume_confirmed and bullish_bias
        short_entry = breakout_down and volume_confirmed and bearish_bias
        
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

</think>
#!/usr/bin/env python3
"""
Experiment #6231: 6h Donchian(20) breakout + 1d Weekly Pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R1/S1) capture institutional 
flow while avoiding false breakouts. Weekly pivots provide key support/resistance from larger 
participants. Volume >1.5x average confirms participation. Designed for 6h timeframe to target 
50-150 trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
Works in both bull/bear by using pivot direction as trend filter and Donchian for breakout timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6231_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # === HTF: 1d data for Weekly Pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot points from prior week's OHLC
        # Need at least 5 days (1 week) of data
        weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
        weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().shift(1)
        weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).mean().shift(1)
        
        # Weekly Pivot Point (PP) = (H + L + C) / 3
        pp = (weekly_high + weekly_low + weekly_close) / 3.0
        # R1 = (2 * PP) - L, S1 = (2 * PP) - H
        r1 = (2 * pp) - weekly_low
        s1 = (2 * pp) - weekly_high
        
        # Align to 6h timeframe
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    else:
        pp_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 5) + 1  # Donchian, volume avg, ATR, weekly lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21:00-23:59 UTC) ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5  # Volume filter for stronger signals
        
        # Weekly Pivot direction filter: price relative to PP
        bullish_bias = price > pp_aligned[i]
        bearish_bias = price < pp_aligned[i]
        
        # Entry conditions: breakout with volume AND pivot alignment
        # Long: breakout up with volume AND price above PP (bullish bias)
        # Short: breakout down with volume AND price below PP (bearish bias)
        long_entry = breakout_up and volume_confirmed and bullish_bias
        short_entry = breakout_down and volume_confirmed and bearish_bias
        
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