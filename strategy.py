#!/usr/bin/env python3
"""
Experiment #3267: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly pivot direction (from 1d HTF) capture institutional flow with low trade frequency. Volume spike (>2.0x 20-period average) confirms breakout strength. ATR trailing stop (2.5x) manages risk. Position size 0.25. Target: 75-150 total trades over 4 years (19-37/year). Works in bull markets via trend continuation and bear markets via mean reversion from extremes using price channels and volatility filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3267_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Weekly high = max of high over last 5 trading days (approx 5*24/6 = 20 of 6h bars, but we use 1d)
    # We'll use the prior 1d bar's high/low/close for daily pivot, then derive weekly from 5-day aggregation
    # Simplified: use prior 1d bar to calculate daily pivot, then weekly bias from 5-day EMA of pivots
    # But for true weekly pivot, we need to aggregate to weekly first. Instead, we use:
    # Weekly pivot = (prior weekly high + prior weekly low + prior weekly close) / 3
    # Since we don't have weekly data directly, we approximate using 1d data over 5-day window
    # Prior week's OHLC: we look back 5 days (5 * 24/6 = 20 of our 6h bars, but in 1d terms: 5 days)
    # We'll calculate rolling weekly OHLC on 1d data
    
    # Calculate prior week's high, low, close on 1d data (using 5-day lookback, excluding current day)
    # We use shift(1) to ensure we only use prior completed week
    lookback_days = 5
    # For each point, we want the high/low/close from the prior 5-day period (week)
    # We'll use rolling window of 5 on 1d data, then shift by 1 to exclude current day
    if len(high_1d) >= lookback_days:
        # Prior week's high = max of high over prior 5 days
        prior_week_high = pd.Series(high_1d).rolling(window=lookback_days, min_periods=lookback_days).max().shift(1).values
        # Prior week's low = min of low over prior 5 days
        prior_week_low = pd.Series(low_1d).rolling(window=lookback_days, min_periods=lookback_days).min().shift(1).values
        # Prior week's close = close of the prior day (5 days ago? Actually, weekly close is Friday's close)
        # Simplified: use close from 5 days ago as proxy for prior week's close
        prior_week_close = pd.Series(close_1d).shift(lookback_days).values  # close 5 days ago
    else:
        prior_week_high = np.full_like(high_1d, np.nan)
        prior_week_low = np.full_like(low_1d, np.nan)
        prior_week_close = np.full_like(close_1d, np.nan)
    
    # Weekly pivot point = (prior_week_high + prior_week_low + prior_week_close) / 3
    weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    
    # Weekly support/resistance levels (simplified: using pivot formula)
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # We'll use the bias: price > weekly_pivot = bullish bias, price < weekly_pivot = bearish bias
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
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
    
    warmup = max(50, lookback, 20, 14, lookback_days)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Weekly pivot bias: long above weekly pivot, short below weekly pivot
            price_vs_pivot = price - weekly_pivot_aligned[i]
            
            # Long entry: price breaks above Donchian high with bullish weekly pivot bias
            if price > highest_high[i] and price_vs_pivot > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with bearish weekly pivot bias
            elif price < lowest_low[i] and price_vs_pivot < 0:
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
Experiment #3267: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly pivot direction (from 1d HTF) capture institutional flow with low trade frequency. Volume spike (>2.0x 20-period average) confirms breakout strength. ATR trailing stop (2.5x) manages risk. Position size 0.25. Target: 75-150 total trades over 4 years (19-37/year). Works in bull markets via trend continuation and bear markets via mean reversion from extremes using price channels and volatility filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3267_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Weekly high = max of high over last 5 trading days (approx 5*24/6 = 20 of 6h bars, but we use 1d)
    # We'll use the prior 1d bar's high/low/close for daily pivot, then derive weekly from 5-day aggregation
    # Simplified: use prior 1d bar to calculate daily pivot, then weekly bias from 5-day EMA of pivots
    # But for true weekly pivot, we need to aggregate to weekly first. Instead, we use:
    # Weekly pivot = (prior weekly high + prior weekly low + prior weekly close) / 3
    # Since we don't have weekly data directly, we approximate using 1d data over 5-day window
    # Prior week's OHLC: we look back 5 days (5 * 24/6 = 20 of our 6h bars, but in 1d terms: 5 days)
    # We'll calculate rolling weekly OHLC on 1d data
    
    # Calculate prior week's high, low, close on 1d data (using 5-day lookback, excluding current day)
    # We use shift(1) to ensure we only use prior completed week
    lookback_days = 5
    # For each point, we want the high/low/close from the prior 5-day period (week)
    # We'll use rolling window of 5 on 1d data, then shift by 1 to exclude current day
    if len(high_1d) >= lookback_days:
        # Prior week's high = max of high over prior 5 days
        prior_week_high = pd.Series(high_1d).rolling(window=lookback_days, min_periods=lookback_days).max().shift(1).values
        # Prior week's low = min of low over prior 5 days
        prior_week_low = pd.Series(low_1d).rolling(window=lookback_days, min_periods=lookback_days).min().shift(1).values
        # Prior week's close = close of the prior day (5 days ago? Actually, weekly close is Friday's close)
        # Simplified: use close from 5 days ago as proxy for prior week's close
        prior_week_close = pd.Series(close_1d).shift(lookback_days).values  # close 5 days ago
    else:
        prior_week_high = np.full_like(high_1d, np.nan)
        prior_week_low = np.full_like(low_1d, np.nan)
        prior_week_close = np.full_like(close_1d, np.nan)
    
    # Weekly pivot point = (prior_week_high + prior_week_low + prior_week_close) / 3
    weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    
    # Weekly support/resistance levels (simplified: using pivot formula)
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # We'll use the bias: price > weekly_pivot = bullish bias, price < weekly_pivot = bearish bias
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
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
    
    warmup = max(50, lookback, 20, 14, lookback_days)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Weekly pivot bias: long above weekly pivot, short below weekly pivot
            price_vs_pivot = price - weekly_pivot_aligned[i]
            
            # Long entry: price breaks above Donchian high with bullish weekly pivot bias
            if price > highest_high[i] and price_vs_pivot > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with bearish weekly pivot bias
            elif price < lowest_low[i] and price_vs_pivot < 0:
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