#!/usr/bin/env python3
"""
Experiment #3927: 6h Donchian(20) breakout + 1d/1w pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with daily/weekly pivot levels capture medium-term momentum with confluence. 
Daily pivot direction determines bias (long above daily pivot, short below). Weekly R4/S4 levels act as breakout/continuation zones. 
Volume > 2.0x MA(20) confirms breakout strength. ATR(14) trailing stop (2.0x) manages risk. 
Designed for both bull/bear: pivot adapts to regimes, Donchian captures breakouts, volume filters noise.
Target: 75-150 trades over 4 years (19-38/year). Uses discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3927_6h_donchian20_1d_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for daily pivot and EMA filter ===
    df_1d = get_htf_data(prices, '1d')
    # Daily pivot calculation (standard floor trader pivots)
    pivots_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    r1_1d = 2 * pivots_1d - df_1d['low']
    s1_1d = 2 * pivots_1d - df_1d['high']
    r2_1d = pivots_1d + (df_1d['high'] - df_1d['low'])
    s2_1d = pivots_1d - (df_1d['high'] - df_1d['low'])
    r3_1d = df_1d['high'] + 2 * (pivots_1d - df_1d['low'])
    s3_1d = df_1d['low'] - 2 * (df_1d['high'] - pivots_1d)
    # Weekly data for R4/S4 breakout levels
    df_1w = get_htf_data(prices, '1w')
    pivots_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    r4_1w = pivots_1w + 3 * (df_1w['high'] - df_1w['low'])  # R4 = PP + 3*(H-L)
    s4_1w = pivots_1w - 3 * (df_1w['high'] - df_1w['low'])  # S4 = PP - 3*(H-L)
    
    # Align HTF data to LTF
    pivots_1d_aligned = align_htf_to_ltf(prices, df_1d, pivots_1d.values)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d.values)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d.values)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d.values)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d.values)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d.values)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d.values)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w.values)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w.values)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
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
    
    warmup = max(lookback_dc + 1, 20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pivots_1d_aligned[i]) or np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
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
                # Exit if price breaks below daily S1 (pivot support)
                elif price < s1_1d_aligned[i]:
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
                # Exit if price breaks above daily R1 (pivot resistance)
                elif price > r1_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) to filter noise
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Determine bias from daily pivot: bullish above pivot, bearish below
            bullish_bias = price > pivots_1d_aligned[i]
            bearish_bias = price < pivots_1d_aligned[i]
            
            # Long entry: breakout above weekly R4 OR daily R3 with bullish bias
            long_breakout = (price > r4_1w_aligned[i] or price > r3_1d_aligned[i]) and bullish_bias
            # Short entry: breakdown below weekly S4 OR daily S3 with bearish bias
            short_breakout = (price < s4_1w_aligned[i] or price < s3_1d_aligned[i]) and bearish_bias
            
            if long_breakout and not short_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_breakout and not long_breakout:
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
Experiment #3927: 6h Donchian(20) breakout + 1d/1w pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with daily/weekly pivot levels capture medium-term momentum with confluence. 
Daily pivot direction determines bias (long above daily pivot, short below). Weekly R4/S4 levels act as breakout/continuation zones. 
Volume > 2.0x MA(20) confirms breakout strength. ATR(14) trailing stop (2.0x) manages risk. 
Designed for both bull/bear: pivot adapts to regimes, Donchian captures breakouts, volume filters noise.
Target: 75-150 trades over 4 years (19-38/year). Uses discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3927_6h_donchian20_1d_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for daily pivot and EMA filter ===
    df_1d = get_htf_data(prices, '1d')
    # Daily pivot calculation (standard floor trader pivots)
    pivots_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    r1_1d = 2 * pivots_1d - df_1d['low']
    s1_1d = 2 * pivots_1d - df_1d['high']
    r2_1d = pivots_1d + (df_1d['high'] - df_1d['low'])
    s2_1d = pivots_1d - (df_1d['high'] - df_1d['low'])
    r3_1d = df_1d['high'] + 2 * (pivots_1d - df_1d['low'])
    s3_1d = df_1d['low'] - 2 * (df_1d['high'] - pivots_1d)
    # Weekly data for R4/S4 breakout levels
    df_1w = get_htf_data(prices, '1w')
    pivots_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    r4_1w = pivots_1w + 3 * (df_1w['high'] - df_1w['low'])  # R4 = PP + 3*(H-L)
    s4_1w = pivots_1w - 3 * (df_1w['high'] - df_1w['low'])  # S4 = PP - 3*(H-L)
    
    # Align HTF data to LTF
    pivots_1d_aligned = align_htf_to_ltf(prices, df_1d, pivots_1d.values)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d.values)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d.values)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d.values)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d.values)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d.values)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d.values)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w.values)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w.values)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
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
    
    warmup = max(lookback_dc + 1, 20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pivots_1d_aligned[i]) or np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
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
                # Exit if price breaks below daily S1 (pivot support)
                elif price < s1_1d_aligned[i]:
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
                # Exit if price breaks above daily R1 (pivot resistance)
                elif price > r1_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) to filter noise
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Determine bias from daily pivot: bullish above pivot, bearish below
            bullish_bias = price > pivots_1d_aligned[i]
            bearish_bias = price < pivots_1d_aligned[i]
            
            # Long entry: breakout above weekly R4 OR daily R3 with bullish bias
            long_breakout = (price > r4_1w_aligned[i] or price > r3_1d_aligned[i]) and bullish_bias
            # Short entry: breakdown below weekly S4 OR daily S3 with bearish bias
            short_breakout = (price < s4_1w_aligned[i] or price < s3_1d_aligned[i]) and bearish_bias
            
            if long_breakout and not short_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_breakout and not long_breakout:
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