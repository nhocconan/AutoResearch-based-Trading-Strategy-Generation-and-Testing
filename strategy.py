#!/usr/bin/env python3
"""
Experiment #231: 6h Camarilla Pivot Reversal with 1d Volume Spike and ATR Stoploss

HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) calculated from 1d OHLC act as institutional support/resistance. Price reverting from R3/S3 with 1d volume confirmation captures mean-reversion in ranging markets, while breaks of R4/S4 with volume indicate continuation in trending markets. The 6h timeframe reduces noise vs lower TFs while capturing sufficient swings. Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. Works in bull/bear via mean-reversion in ranges and trend-following on breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_pivot_reversal_1d_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    if len(df_1d) >= 1:
        # Camarilla: based on previous day's range
        # R4 = close + ((high - low) * 1.1/2)
        # R3 = close + ((high - low) * 1.1/4)
        # S3 = close - ((high - low) * 1.1/4)
        # S4 = close - ((high - low) * 1.1/2)
        prev_close = df_1d['close'].values
        prev_high = df_1d['high'].values
        prev_low = df_1d['low'].values
        
        # Calculate levels for each 1d bar
        camarilla_r4 = np.full_like(prev_close, np.nan)
        camarilla_r3 = np.full_like(prev_close, np.nan)
        camarilla_s3 = np.full_like(prev_close, np.nan)
        camarilla_s4 = np.full_like(prev_close, np.nan)
        
        for i in range(len(prev_close)):
            if i > 0:  # Use previous day's data (avoid look-ahead)
                rng = prev_high[i-1] - prev_low[i-1]
                camarilla_r4[i] = prev_close[i-1] + (rng * 1.1 / 2)
                camarilla_r3[i] = prev_close[i-1] + (rng * 1.1 / 4)
                camarilla_s3[i] = prev_close[i-1] - (rng * 1.1 / 4)
                camarilla_s4[i] = prev_close[i-1] - (rng * 1.1 / 2)
        
        # Align to 6h timeframe (shift(1) inside align_htf_to_ltf for completed bars)
        r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        r4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Relative to Camarilla Levels ---
        near_r3 = abs(close[i] - r3_aligned[i]) / r3_aligned[i] < 0.005  # Within 0.5% of R3
        near_s3 = abs(close[i] - s3_aligned[i]) / s3_aligned[i] < 0.005  # Within 0.5% of S3
        breaks_r4 = close[i] > r4_aligned[i]
        breaks_s4 = close[i] < s4_aligned[i]
        
        # --- Volume Confirmation (1d volume spike) ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price reaches S3 (mean reversion target) OR breaks below S4 (stop)
                    if close[i] <= s3_aligned[i] or close[i] < s4_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price reaches R3 (mean reversion target) OR breaks above R4 (stop)
                    if close[i] >= r3_aligned[i] or close[i] > r4_aligned[i]:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Mean reversion longs: price near S3 with volume, expect bounce to R3
        if near_s3 and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Mean reversion shorts: price near R3 with volume, expect drop to S3
        elif near_r3 and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        # Breakout longs: price breaks R4 with volume, expect continuation
        elif breaks_r4 and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Breakout shorts: price breaks S4 with volume, expect continuation
        elif breaks_s4 and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #231: 6h Camarilla Pivot Reversal with 1d Volume Spike and ATR Stoploss

HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) calculated from 1d OHLC act as institutional support/resistance. Price reverting from R3/S3 with 1d volume confirmation captures mean-reversion in ranging markets, while breaks of R4/S4 with volume indicate continuation in trending markets. The 6h timeframe reduces noise vs lower TFs while capturing sufficient swings. Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. Works in bull/bear via mean-reversion in ranges and trend-following on breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_pivot_reversal_1d_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    if len(df_1d) >= 1:
        # Camarilla: based on previous day's range
        # R4 = close + ((high - low) * 1.1/2)
        # R3 = close + ((high - low) * 1.1/4)
        # S3 = close - ((high - low) * 1.1/4)
        # S4 = close - ((high - low) * 1.1/2)
        prev_close = df_1d['close'].values
        prev_high = df_1d['high'].values
        prev_low = df_1d['low'].values
        
        # Calculate levels for each 1d bar
        camarilla_r4 = np.full_like(prev_close, np.nan)
        camarilla_r3 = np.full_like(prev_close, np.nan)
        camarilla_s3 = np.full_like(prev_close, np.nan)
        camarilla_s4 = np.full_like(prev_close, np.nan)
        
        for i in range(len(prev_close)):
            if i > 0:  # Use previous day's data (avoid look-ahead)
                rng = prev_high[i-1] - prev_low[i-1]
                camarilla_r4[i] = prev_close[i-1] + (rng * 1.1 / 2)
                camarilla_r3[i] = prev_close[i-1] + (rng * 1.1 / 4)
                camarilla_s3[i] = prev_close[i-1] - (rng * 1.1 / 4)
                camarilla_s4[i] = prev_close[i-1] - (rng * 1.1 / 2)
        
        # Align to 6h timeframe (shift(1) inside align_htf_to_ltf for completed bars)
        r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        r4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Relative to Camarilla Levels ---
        near_r3 = abs(close[i] - r3_aligned[i]) / r3_aligned[i] < 0.005  # Within 0.5% of R3
        near_s3 = abs(close[i] - s3_aligned[i]) / s3_aligned[i] < 0.005  # Within 0.5% of S3
        breaks_r4 = close[i] > r4_aligned[i]
        breaks_s4 = close[i] < s4_aligned[i]
        
        # --- Volume Confirmation (1d volume spike) ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price reaches S3 (mean reversion target) OR breaks below S4 (stop)
                    if close[i] <= s3_aligned[i] or close[i] < s4_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price reaches R3 (mean reversion target) OR breaks above R4 (stop)
                    if close[i] >= r3_aligned[i] or close[i] > r4_aligned[i]:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Mean reversion longs: price near S3 with volume, expect bounce to R3
        if near_s3 and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Mean reversion shorts: price near R3 with volume, expect drop to S3
        elif near_r3 and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        # Breakout longs: price breaks R4 with volume, expect continuation
        elif breaks_r4 and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Breakout shorts: price breaks S4 with volume, expect continuation
        elif breaks_s4 and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals