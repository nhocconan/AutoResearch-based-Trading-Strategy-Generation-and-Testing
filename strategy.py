#!/usr/bin/env python3
"""
Experiment #2531: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian channel breakouts on 6h timeframe aligned with weekly pivot levels from 1d data
capture institutional participation during trend acceleration. Weekly pivot acts as regime filter:
price above weekly pivot = bullish bias (long breakouts), price below weekly pivot = bearish bias (short breakouts).
Volume confirmation ensures breakout validity. Designed for 50-150 total trades over 4 years (12-37/year)
to minimize fee drag while maintaining statistical significance. Works in both bull and bear markets
by using pivot direction for bias and Donchian breakouts for entry timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2531_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot from prior week's 1d OHLC (using last 5 trading days)
    # We need to group 1d data into weeks and calculate pivot for prior week
    # Since we don't have explicit week grouping, we'll use rolling window of 5 days
    # and calculate pivot from the 5-day period ending 1 day ago (to avoid look-ahead)
    if len(close_1d) >= 6:  # Need at least 6 days: 5 for calculation + 1 for shift
        # Calculate rolling high, low, close over 5 days
        rolling_high_5 = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        rolling_low_5 = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        rolling_close_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot formula: (High + Low + Close) / 3
        weekly_pivot = (rolling_high_5 + rolling_low_5 + rolling_close_5) / 3.0
        
        # Shift by 1 to use prior week's pivot (avoid look-ahead)
        weekly_pivot = np.roll(weekly_pivot, 1)
        weekly_pivot[0] = np.nan  # First value is invalid after roll
        
        # Bias: 1 if price above weekly pivot (bullish), -1 if below (bearish)
        weekly_bias = np.where(close_1d > weekly_pivot, 1, -1)
        weekly_bias[np.isnan(weekly_pivot)] = 0  # No bias when pivot unavailable
        
        # Align to 6h timeframe
        weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    else:
        weekly_bias_aligned = np.full(n, 0.0)
    
    # === 6h Indicators: Donchian(20) channels, Volume MA(20) ===
    # Donchian channels (20-period high/low)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 60  # sufficient for 20-period indicators + 5-day weekly pivot
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_bias_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry (using Donchian width as ATR proxy)
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15  # approximate ATR from channel width
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian low (mean reversion)
                elif price < lowest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian high (mean reversion)
                elif price > highest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly pivot bias for directional filter
        bias = weekly_bias_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and bias != 0:
            # Long entry: price breaks above Donchian high with bullish bias
            if bias > 0 and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with bearish bias
            elif bias < 0 and price < lowest_20[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #2531: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian channel breakouts on 6h timeframe aligned with weekly pivot levels from 1d data
capture institutional participation during trend acceleration. Weekly pivot acts as regime filter:
price above weekly pivot = bullish bias (long breakouts), price below weekly pivot = bearish bias (short breakouts).
Volume confirmation ensures breakout validity. Designed for 50-150 total trades over 4 years (12-37/year)
to minimize fee drag while maintaining statistical significance. Works in both bull and bear markets
by using pivot direction for bias and Donchian breakouts for entry timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2531_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot from prior week's 1d OHLC (using last 5 trading days)
    # We need to group 1d data into weeks and calculate pivot for prior week
    # Since we don't have explicit week grouping, we'll use rolling window of 5 days
    # and calculate pivot from the 5-day period ending 1 day ago (to avoid look-ahead)
    if len(close_1d) >= 6:  # Need at least 6 days: 5 for calculation + 1 for shift
        # Calculate rolling high, low, close over 5 days
        rolling_high_5 = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        rolling_low_5 = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        rolling_close_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot formula: (High + Low + Close) / 3
        weekly_pivot = (rolling_high_5 + rolling_low_5 + rolling_close_5) / 3.0
        
        # Shift by 1 to use prior week's pivot (avoid look-ahead)
        weekly_pivot = np.roll(weekly_pivot, 1)
        weekly_pivot[0] = np.nan  # First value is invalid after roll
        
        # Bias: 1 if price above weekly pivot (bullish), -1 if below (bearish)
        weekly_bias = np.where(close_1d > weekly_pivot, 1, -1)
        weekly_bias[np.isnan(weekly_pivot)] = 0  # No bias when pivot unavailable
        
        # Align to 6h timeframe
        weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    else:
        weekly_bias_aligned = np.full(n, 0.0)
    
    # === 6h Indicators: Donchian(20) channels, Volume MA(20) ===
    # Donchian channels (20-period high/low)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 60  # sufficient for 20-period indicators + 5-day weekly pivot
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_bias_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry (using Donchian width as ATR proxy)
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15  # approximate ATR from channel width
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian low (mean reversion)
                elif price < lowest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian high (mean reversion)
                elif price > highest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly pivot bias for directional filter
        bias = weekly_bias_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and bias != 0:
            # Long entry: price breaks above Donchian high with bullish bias
            if bias > 0 and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with bearish bias
            elif bias < 0 and price < lowest_20[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals