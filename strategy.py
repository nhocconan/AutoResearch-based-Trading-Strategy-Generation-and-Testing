#!/usr/bin/env python3
"""
Experiment #4631: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h price breaking Donchian(20) channels from prior 4 weeks with weekly pivot filter (price > weekly pivot for longs, < weekly pivot for shorts) and volume confirmation (>1.5x avg volume) captures strong momentum breakouts with directional bias from higher timeframe. Weekly pivot acts as regime filter to avoid counter-trend trades. Uses 1d HTF for Donchian and weekly pivot calculation to avoid look-ahead. Discrete sizing (0.25) and ATR trailing stop (2.0x) manage risk. Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4631_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Donchian channels and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian(20) from prior 20 days OHLC (shifted by 1 to avoid look-ahead)
    if len(df_1d) >= 20:
        # Use prior 20 days' high/low (shifted by 1)
        ph = np.concatenate([[np.nan] * 20, df_1d['high'].values[:-20]])  # prior 20 days high
        pl = np.concatenate([[np.nan] * 20, df_1d['low'].values[:-20]])   # prior 20 days low
        
        # Rolling max/min of prior 20 days
        donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Calculate weekly pivot from prior week OHLC (shifted by 1 to avoid look-ahead)
    # Weekly pivot = (prior week high + prior week low + prior week close) / 3
    if len(df_1d) >= 5:
        # Get prior week's OHLC (5 trading days ago)
        weekly_high = np.concatenate([[np.nan] * 5, df_1d['high'].values[:-5]])   # prior week high
        weekly_low = np.concatenate([[np.nan] * 5, df_1d['low'].values[:-5]])     # prior week low
        weekly_close = np.concatenate([[np.nan] * 5, df_1d['close'].values[:-5]]) # prior week close
        
        # Weekly pivot = (H+L+C)/3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    else:
        weekly_pivot = np.full(n, np.nan)
    
    # Align HTF levels to 6h timeframe
    if len(donchian_high) > 0:
        dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
        dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
        wp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        dh_aligned = np.full(n, np.nan)
        dl_aligned = np.full(n, np.nan)
        wp_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 14)  # Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(wp_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation for breakouts (>1.5x)
        vol_breakout = vol_ratio[i] > 1.5
        
        # Weekly pivot filter: price > pivot for longs, price < pivot for shorts
        pivot_long_filter = price > wp_aligned[i]
        pivot_short_filter = price < wp_aligned[i]
        
        # Breakout conditions: price breaks Donchian high/low with volume and pivot confirmation
        breakout_long = price > dh_aligned[i] and vol_breakout and pivot_long_filter
        breakout_short = price < dl_aligned[i] and vol_breakout and pivot_short_filter
        
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
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
Experiment #4631: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h price breaking Donchian(20) channels from prior 4 weeks with weekly pivot filter (price > weekly pivot for longs, < weekly pivot for shorts) and volume confirmation (>1.5x avg volume) captures strong momentum breakouts with directional bias from higher timeframe. Weekly pivot acts as regime filter to avoid counter-trend trades. Uses 1d HTF for Donchian and weekly pivot calculation to avoid look-ahead. Discrete sizing (0.25) and ATR trailing stop (2.0x) manage risk. Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4631_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Donchian channels and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian(20) from prior 20 days OHLC (shifted by 1 to avoid look-ahead)
    if len(df_1d) >= 20:
        # Use prior 20 days' high/low (shifted by 1)
        ph = np.concatenate([[np.nan] * 20, df_1d['high'].values[:-20]])  # prior 20 days high
        pl = np.concatenate([[np.nan] * 20, df_1d['low'].values[:-20]])   # prior 20 days low
        
        # Rolling max/min of prior 20 days
        donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Calculate weekly pivot from prior week OHLC (shifted by 1 to avoid look-ahead)
    # Weekly pivot = (prior week high + prior week low + prior week close) / 3
    if len(df_1d) >= 5:
        # Get prior week's OHLC (5 trading days ago)
        weekly_high = np.concatenate([[np.nan] * 5, df_1d['high'].values[:-5]])   # prior week high
        weekly_low = np.concatenate([[np.nan] * 5, df_1d['low'].values[:-5]])     # prior week low
        weekly_close = np.concatenate([[np.nan] * 5, df_1d['close'].values[:-5]]) # prior week close
        
        # Weekly pivot = (H+L+C)/3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    else:
        weekly_pivot = np.full(n, np.nan)
    
    # Align HTF levels to 6h timeframe
    if len(donchian_high) > 0:
        dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
        dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
        wp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        dh_aligned = np.full(n, np.nan)
        dl_aligned = np.full(n, np.nan)
        wp_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 14)  # Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(wp_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation for breakouts (>1.5x)
        vol_breakout = vol_ratio[i] > 1.5
        
        # Weekly pivot filter: price > pivot for longs, price < pivot for shorts
        pivot_long_filter = price > wp_aligned[i]
        pivot_short_filter = price < wp_aligned[i]
        
        # Breakout conditions: price breaks Donchian high/low with volume and pivot confirmation
        breakout_long = price > dh_aligned[i] and vol_breakout and pivot_long_filter
        breakout_short = price < dl_aligned[i] and vol_breakout and pivot_short_filter
        
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals