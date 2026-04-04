#!/usr/bin/env python3
"""
Experiment #4347: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian breakouts on 6h capture momentum when aligned with weekly pivot bias (price above/below weekly pivot) and confirmed by volume spikes (>1.5x average). Weekly pivot provides structural bias from longer timeframe, reducing false breakouts in ranging markets. Volume confirmation ensures institutional participation. Designed for 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4347_6h_donchian20_1w_pivot_vol_v1"
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
    
    # === Precompute HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week (Mon-Sun)
        # We need to group daily data into weeks (Monday start)
        # Create a series of week start dates (Monday)
        dates_1d = pd.DatetimeIndex(df_1d['open_time'].values)
        # Week start: Monday of each week
        week_start = dates_1d - pd.to_timedelta(dates_1d.weekday, unit='D')
        
        # Group by week start and calculate weekly OHLC
        weekly_high = pd.Series(df_1d['high'].values, index=dates_1d).groupby(week_start).max()
        weekly_low = pd.Series(df_1d['low'].values, index=dates_1d).groupby(week_start).min()
        weekly_close = pd.Series(df_1d['close'].values, index=dates_1d).groupby(week_start).last()
        
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Align weekly pivot to 1d data (forward fill within week)
        weekly_pivot_1d = pd.Series(index=dates_1d, dtype=np.float64)
        for ws in weekly_pivot.index:
            mask = (dates_1d >= ws) & (dates_1d < ws + pd.Timedelta(days=7))
            weekly_pivot_1d.loc[mask] = weekly_pivot[ws]
        
        weekly_pivot_1d_array = weekly_pivot_1d.values
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_1d_array)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
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
    
    warmup = max(lookback, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i])):
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
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        # Weekly pivot bias: price above pivot = bullish bias, below = bearish bias
        bullish_bias = price > weekly_pivot_aligned[i]
        bearish_bias = price < weekly_pivot_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high[i] > highest_high[i-1]  # New 20-period high
        breakout_down = low[i] < lowest_low[i-1]   # New 20-period low
        
        # Long conditions: bullish bias + upside breakout + volume
        long_entry = bullish_bias and breakout_up and volume_confirm
        
        # Short conditions: bearish bias + downside breakout + volume
        short_entry = bearish_bias and breakout_down and volume_confirm
        
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
Experiment #4347: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian breakouts on 6h capture momentum when aligned with weekly pivot bias (price above/below weekly pivot) and confirmed by volume spikes (>1.5x average). Weekly pivot provides structural bias from longer timeframe, reducing false breakouts in ranging markets. Volume confirmation ensures institutional participation. Designed for 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4347_6h_donchian20_1w_pivot_vol_v1"
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
    
    # === Precompute HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week (Mon-Sun)
        # We need to group daily data into weeks (Monday start)
        # Create a series of week start dates (Monday)
        dates_1d = pd.DatetimeIndex(df_1d['open_time'].values)
        # Week start: Monday of each week
        week_start = dates_1d - pd.to_timedelta(dates_1d.weekday, unit='D')
        
        # Group by week start and calculate weekly OHLC
        weekly_high = pd.Series(df_1d['high'].values, index=dates_1d).groupby(week_start).max()
        weekly_low = pd.Series(df_1d['low'].values, index=dates_1d).groupby(week_start).min()
        weekly_close = pd.Series(df_1d['close'].values, index=dates_1d).groupby(week_start).last()
        
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Align weekly pivot to 1d data (forward fill within week)
        weekly_pivot_1d = pd.Series(index=dates_1d, dtype=np.float64)
        for ws in weekly_pivot.index:
            mask = (dates_1d >= ws) & (dates_1d < ws + pd.Timedelta(days=7))
            weekly_pivot_1d.loc[mask] = weekly_pivot[ws]
        
        weekly_pivot_1d_array = weekly_pivot_1d.values
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_1d_array)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
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
    
    warmup = max(lookback, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i])):
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
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        # Weekly pivot bias: price above pivot = bullish bias, below = bearish bias
        bullish_bias = price > weekly_pivot_aligned[i]
        bearish_bias = price < weekly_pivot_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high[i] > highest_high[i-1]  # New 20-period high
        breakout_down = low[i] < lowest_low[i-1]   # New 20-period low
        
        # Long conditions: bullish bias + upside breakout + volume
        long_entry = bullish_bias and breakout_up and volume_confirm
        
        # Short conditions: bearish bias + downside breakout + volume
        short_entry = bearish_bias and breakout_down and volume_confirm
        
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