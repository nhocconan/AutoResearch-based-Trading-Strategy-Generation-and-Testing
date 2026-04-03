#!/usr/bin/env python3
"""
Experiment #619: 6h Donchian(20) breakout + 12h Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Camarilla pivot structure (R3/S3 for mean reversion, R4/S4 for breakout) capture institutional order flow with reduced whipsaw. Volume confirmation ensures participation. Target: 50-150 total trades over 4 years via tight entry conditions (Donchian breakout + HTF pivot level + volume spike).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_619_6h_donchian20_12h_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for previous 12h bar
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We use the previous completed 12h bar for pivot calculation
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r4 = pivot + (range_12h * 1.1 / 2.0)
    r3 = pivot + (range_12h * 1.1 / 4.0)
    s3 = pivot - (range_12h * 1.1 / 4.0)
    s4 = pivot - (range_12h * 1.1 / 2.0)
    
    # Align HTF Camarilla levels to 6h timeframe (shifted by 1 for completed bars only)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian and pivot calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Camarilla Pivot Logic ---
        # Fade at R3/S3 (mean reversion in range)
        # Breakout continuation at R4/S4 (strong momentum)
        at_r3 = abs(price - r3_aligned[i]) / r3_aligned[i] < 0.002  # within 0.2% of R3
        at_s3 = abs(price - s3_aligned[i]) / s3_aligned[i] < 0.002  # within 0.2% of S3
        above_r4 = price > r4_aligned[i]
        below_s4 = price < s4_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~36h on 6h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + (above R4 for continuation OR at S3 for mean reversion bounce)
            if breakout_up and (above_r4 or at_s3):
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + (below S4 for continuation OR at R3 for mean reversion fade)
            elif breakout_down and (below_s4 or at_r3):
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #619: 6h Donchian(20) breakout + 12h Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Camarilla pivot structure (R3/S3 for mean reversion, R4/S4 for breakout) capture institutional order flow with reduced whipsaw. Volume confirmation ensures participation. Target: 50-150 total trades over 4 years via tight entry conditions (Donchian breakout + HTF pivot level + volume spike).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_619_6h_donchian20_12h_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for previous 12h bar
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We use the previous completed 12h bar for pivot calculation
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r4 = pivot + (range_12h * 1.1 / 2.0)
    r3 = pivot + (range_12h * 1.1 / 4.0)
    s3 = pivot - (range_12h * 1.1 / 4.0)
    s4 = pivot - (range_12h * 1.1 / 2.0)
    
    # Align HTF Camarilla levels to 6h timeframe (shifted by 1 for completed bars only)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian and pivot calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Camarilla Pivot Logic ---
        # Fade at R3/S3 (mean reversion in range)
        # Breakout continuation at R4/S4 (strong momentum)
        at_r3 = abs(price - r3_aligned[i]) / r3_aligned[i] < 0.002  # within 0.2% of R3
        at_s3 = abs(price - s3_aligned[i]) / s3_aligned[i] < 0.002  # within 0.2% of S3
        above_r4 = price > r4_aligned[i]
        below_s4 = price < s4_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~36h on 6h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + (above R4 for continuation OR at S3 for mean reversion bounce)
            if breakout_up and (above_r4 or at_s3):
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + (below S4 for continuation OR at R3 for mean reversion fade)
            elif breakout_down and (below_s4 or at_r3):
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals