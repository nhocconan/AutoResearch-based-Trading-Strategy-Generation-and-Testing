#!/usr/bin/env python3
"""
Experiment #119: 6h Donchian(20) breakout + 12h Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) with volume confirmation capture institutional order flow. Works in bull/bear via pivot structure that adapts to volatility. Uses discrete sizing (0.25) and ATR stoploss (2.0*ATR). Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_119_6h_donchian20_12h_camarilla_vol_v1"
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
    # Calculate Camarilla levels from previous 12h bar (OHLC)
    # Camarilla: based on previous day's range, but we use previous 12h bar
    prev_close_12h = df_12h['close'].shift(1).values  # previous bar close
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    
    # Typical price for pivot calculation
    typical_price = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    range_12h = prev_high_12h - prev_low_12h
    
    # Camarilla levels
    r3 = typical_price + range_12h * 1.1 / 4
    s3 = typical_price - range_12h * 1.1 / 4
    r4 = typical_price + range_12h * 1.1 / 2
    s4 = typical_price - range_12h * 1.1 / 2
    
    # Align to 6h timeframe (shifted by 1 for completed bar)
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
            np.isnan(vol_ratio[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Camarilla Logic ---
        # Near R3/S3: mean reversion (fade)
        # Near R4/S4: breakout continuation
        near_r3 = abs(price - r3_aligned[i]) / price < 0.005  # within 0.5%
        near_s3 = abs(price - s3_aligned[i]) / price < 0.005
        near_r4 = abs(price - r4_aligned[i]) / price < 0.005
        near_s4 = abs(price - s4_aligned[i]) / price < 0.005
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
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
                # If near R4, strong breakout continuation
                if near_r4:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # If near R3, mean reversion fade (only if not breaking out strongly)
                elif near_r3 and not breakout_up:
                    # Actually, if we're near R3 and price is below R3, look for bounce
                    if price < r3_aligned[i]:
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = SIZE
                # Regular breakout above Donchian with volume
                elif breakout_up:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            
            # Short logic
            elif breakout_down:
                # If near S4, strong breakout continuation
                if near_s4:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                # If near S3, mean reversion fade (only if not breaking down strongly)
                elif near_s3 and not breakout_down:
                    # Actually, if we're near S3 and price is above S3, look for bounce down
                    if price > s3_aligned[i]:
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = -SIZE
                # Regular breakdown below Donchian with volume
                elif breakout_down:
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