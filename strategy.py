#!/usr/bin/env python3
"""
Experiment #4619: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h price breaking prior 20-period Donchian channels with volume confirmation (>1.5x avg volume) 
captures strong momentum, filtered by 1d weekly pivot direction (price above/below weekly pivot) 
to align with higher timeframe bias. Uses discrete sizing (0.25) and ATR trailing stop (2.0x). 
Target: 12-37 trades/year on 6h timeframe. Works in bull/bear via pivot filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4619_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot levels from prior 1d OHLC (using prior week's data)
    # Weekly pivot = (Weekly High + Weekly Low + Weekly Close) / 3
    # We approximate by using prior 5 trading days (1 week)
    if len(df_1d) >= 5:
        # Rolling window of 5 days for weekly OHLC
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot levels
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly R1/S1 (standard pivot)
        weekly_r1 = 2 * weekly_pivot - weekly_low
        weekly_s1 = 2 * weekly_pivot - weekly_high
    else:
        weekly_pivot = np.array([])
        weekly_r1 = np.array([])
        weekly_s1 = np.array([])
    
    # Align weekly pivot levels to 6h timeframe
    if len(weekly_pivot) > 0:
        pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    else:
        pivot_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = max(20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x avg volume)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        breakout_long = price > donchian_high[i] and vol_confirm
        breakout_short = price < donchian_low[i] and vol_confirm
        
        # Weekly pivot filter: only trade in direction of higher timeframe bias
        # Long bias: price above weekly pivot and above weekly R1
        # Short bias: price below weekly pivot and below weekly S1
        long_bias = price > pivot_aligned[i] and price > r1_aligned[i]
        short_bias = price < pivot_aligned[i] and price < s1_aligned[i]
        
        # Combine: breakout in direction of weekly pivot bias
        if breakout_long and long_bias:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short and short_bias:
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
Experiment #4619: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h price breaking prior 20-period Donchian channels with volume confirmation (>1.5x avg volume) 
captures strong momentum, filtered by 1d weekly pivot direction (price above/below weekly pivot) 
to align with higher timeframe bias. Uses discrete sizing (0.25) and ATR trailing stop (2.0x). 
Target: 12-37 trades/year on 6h timeframe. Works in bull/bear via pivot filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4619_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot levels from prior 1d OHLC (using prior week's data)
    # Weekly pivot = (Weekly High + Weekly Low + Weekly Close) / 3
    # We approximate by using prior 5 trading days (1 week)
    if len(df_1d) >= 5:
        # Rolling window of 5 days for weekly OHLC
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot levels
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly R1/S1 (standard pivot)
        weekly_r1 = 2 * weekly_pivot - weekly_low
        weekly_s1 = 2 * weekly_pivot - weekly_high
    else:
        weekly_pivot = np.array([])
        weekly_r1 = np.array([])
        weekly_s1 = np.array([])
    
    # Align weekly pivot levels to 6h timeframe
    if len(weekly_pivot) > 0:
        pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    else:
        pivot_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = max(20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x avg volume)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        breakout_long = price > donchian_high[i] and vol_confirm
        breakout_short = price < donchian_low[i] and vol_confirm
        
        # Weekly pivot filter: only trade in direction of higher timeframe bias
        # Long bias: price above weekly pivot and above weekly R1
        # Short bias: price below weekly pivot and below weekly S1
        long_bias = price > pivot_aligned[i] and price > r1_aligned[i]
        short_bias = price < pivot_aligned[i] and price < s1_aligned[i]
        
        # Combine: breakout in direction of weekly pivot bias
        if breakout_long and long_bias:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short and short_bias:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals