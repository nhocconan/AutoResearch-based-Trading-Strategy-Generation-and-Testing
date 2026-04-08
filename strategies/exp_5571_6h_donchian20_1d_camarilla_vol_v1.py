#!/usr/bin/env python3
"""
Experiment #5571: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 1.8x average and aligned 
with daily Camarilla pivot levels (breakout at R4/S4 = continuation, rejection at R3/S3 = mean reversion) 
capture high-probability trend moves in both bull and bear markets. Daily pivot provides structural 
support/resistance from higher timeframe, reducing false breakouts. ATR-based trailing stop limits 
drawdown. Target: 12-37 trades/year (50-150 total over 4 years) with discrete position sizing (0.25) 
to minimize fee drag. Works in bull (breakouts with pivot support) and bear (breakouts with pivot resistance).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5571_6h_donchian20_1d_camarilla_vol_v1"
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
    
    # === HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate Camarilla pivot levels from previous day
        # R4 = C + ((H-L) * 1.1/2)
        # R3 = C + ((H-L) * 1.1/4)
        # S3 = C - ((H-L) * 1.1/4)
        # S4 = C - ((H-L) * 1.1/2)
        # where C = (H+L+Close)/3 (typical price)
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        c_1d = (h_1d + l_1d + df_1d['close'].values) / 3.0
        rng_1d = h_1d - l_1d
        
        r4 = c_1d + (rng_1d * 1.1 / 2.0)
        r3 = c_1d + (rng_1d * 1.1 / 4.0)
        s3 = c_1d - (rng_1d * 1.1 / 4.0)
        s4 = c_1d - (rng_1d * 1.1 / 2.0)
        
        # Align to LTF (6h) with shift(1) for completed bars only
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        # Neutral values if insufficient data
        r4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 2)  # Donchian, volume avg, ATR, HTF warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR Donchian lower band break
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR Donchian upper band break
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8
        
        # Determine bias from daily Camarilla levels
        # Long: breakout above Donchian high with volume AND price > R4 (continuation)
        # Short: breakout below Donchian low with volume AND price < S4 (continuation)
        # Mean reversion: fade at R3/S3 (not used in this version - continuation only)
        long_breakout = breakout_up and volume_confirmed and (price > r4_aligned[i])
        short_breakout = breakout_down and volume_confirmed and (price < s4_aligned[i])
        
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals