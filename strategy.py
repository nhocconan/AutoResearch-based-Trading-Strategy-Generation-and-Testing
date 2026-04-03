#!/usr/bin/env python3
"""
Experiment #099: 6h Donchian(20) breakout + 12h pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h weekly pivot levels (R4/S4 for continuation, R3/S3 for mean reversion) and volume confirmation (>1.5x) captures institutional breakouts while filtering noise. Uses discrete sizing (0.25) and ATR stoploss (2.5*ATR). Target: 75-150 total trades over 4 years (19-38/year). Works in bull (continuation at R4/S4) and bear (mean reversion at R3/S3) via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_099_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    pivot_high = df_12h['high'].values
    pivot_low = df_12h['low'].values
    pivot_close = df_12h['close'].values
    
    # Calculate weekly pivot points (using prior 12h bar's HLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    pivot = (pivot_high + pivot_low + pivot_close) / 3.0
    r1 = 2 * pivot - pivot_low
    s1 = 2 * pivot - pivot_high
    r2 = pivot + (pivot_high - pivot_low)
    s2 = pivot - (pivot_high - pivot_low)
    r3 = pivot_high + 2 * (pivot - pivot_low)
    s3 = pivot_low - 2 * (pivot_high - pivot)
    r4 = r3 + (pivot_high - pivot_low)
    s4 = s3 - (pivot_high - pivot_low)
    
    # Align pivot levels to 6h timeframe (shifted by 1 for completed bar)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
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
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Pivot Level Conditions ---
        # Near R3/S3: mean reversion zone (fade)
        # Near R4/S4: breakout continuation zone (breakout)
        near_r3 = abs(price - r3_aligned[i]) < (0.5 * atr[i])
        near_s3 = abs(price - s3_aligned[i]) < (0.5 * atr[i])
        near_r4 = abs(price - r4_aligned[i]) < (0.5 * atr[i])
        near_s4 = abs(price - s4_aligned[i]) < (0.5 * atr[i])
        
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
            # Long logic
            if breakout_up:
                # Continuation breakout: near R4 or above R4
                if near_r4 or price > r4_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Mean reversion bounce: near S3 with bullish momentum
                elif near_s3 and price > s3_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
            # Short logic
            elif breakout_down:
                # Continuation breakdown: near S4 or below S4
                elif near_s4 or price < s4_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                # Mean reversion bounce: near R3 with bearish momentum
                elif near_r3 and price < r3_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

if __name__ == "__main__":
    # Quick test for syntax errors
    import pandas as pd
    import numpy as np
    print("Strategy loaded successfully")