#!/usr/bin/env python3
"""
Experiment #5359: 6h Donchian(20) breakout + 12h Camarilla pivot levels + volume confirmation
HYPOTHESIS: On 6h timeframe, price breaking above/below the 20-period Donchian channel 
with volume > 2.0x average and aligned with 12h Camarilla pivot levels (breakout at R4/S4, 
fade at R3/S3) captures institutional order flow. The Camarilla levels provide mathematical 
support/resistance based on prior day's range, working in both bull (breakouts) and bear 
markets (fades from overbought/oversold levels). Discrete position sizing (0.25) and 
ATR-based trailing stoploss (2.0x ATR) control risk. Target: 12-37 trades/year (50-150 
total over 4 years) to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5359_6h_donchian20_12h_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 12h data for Camarilla pivot levels ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 2:
        # Calculate Camarilla levels from prior 12h bar's range
        # Camarilla: Based on previous period's high, low, close
        prev_high = df_12h['high'].shift(1).values
        prev_low = df_12h['low'].shift(1).values
        prev_close = df_12h['close'].shift(1).values
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        
        # Camarilla levels
        r3 = pivot + (range_hl * 1.1 / 4.0)
        r4 = pivot + (range_hl * 1.1 / 2.0)
        s3 = pivot - (range_hl * 1.1 / 4.0)
        s4 = pivot - (range_hl * 1.1 / 2.0)
        
        # Store levels for alignment
        camarilla_levels = np.column_stack([r3, r4, s3, s4])
    else:
        camarilla_levels = np.full((n, 4), np.nan)
    
    # Align to LTF (6h) with shift(1) for completed bars only
    camarilla_aligned = align_htf_to_ltf(prices, df_12h, camarilla_levels) if len(df_12h) >= 2 else np.full((n, 4), np.nan)
    
    # Extract individual levels
    r3_aligned = camarilla_aligned[:, 0] if camarilla_aligned.ndim > 1 else np.full(n, np.nan)
    r4_aligned = camarilla_aligned[:, 1] if camarilla_aligned.ndim > 1 else np.full(n, np.nan)
    s3_aligned = camarilla_aligned[:, 2] if camarilla_aligned.ndim > 1 else np.full(n, np.nan)
    s4_aligned = camarilla_aligned[:, 3] if camarilla_aligned.ndim > 1 else np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
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
    
    warmup = max(20, 20, 14) + 1  # Donchian, volume avg, ATR + 1 for HTF shift
    
    for i in range(warmup, n):
        # --- Session Filter: Trade during liquid sessions ---
        hour = hours[i]
        # Focus on major sessions: 00-06 UTC (Asia), 07-12 UTC (Europe), 13-20 UTC (US)
        # Avoid 21-23 UTC (low liquidity between sessions)
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or Camarilla level reversal ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.0 * ATR below highest since entry
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price crosses below S3 (mean reversion level)
                if price <= stop_price or price <= donchian_low[i] or price < s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.0 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price crosses above R3 (mean reversion level)
                if price >= stop_price or price >= donchian_high[i] or price > r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 2.0x average volume (strict)
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Camarilla pivot logic
        # Long: Breakout above R4 (strong bullish) OR fade from S3 (oversold bounce)
        long_breakout = breakout_up and price > r4_aligned[i-1]
        long_fade = price < s3_aligned[i-1] and price > s4_aligned[i-1] and volume_confirmed
        
        # Short: Breakdown below S4 (strong bearish) OR fade from R3 (overbought rejection)
        short_breakout = breakout_down and price < s4_aligned[i-1]
        short_fade = price > r3_aligned[i-1] and price < r4_aligned[i-1] and volume_confirmed
        
        # Entry conditions
        if (long_breakout or long_fade) and volume_confirmed:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif (short_breakout or short_fade) and volume_confirmed:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals