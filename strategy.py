#!/usr/bin/env python3
"""
Experiment #5739: 6h Donchian(20) breakout + 12h Camarilla pivot levels + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 2.0x average and aligned 
with 12h Camarilla pivot levels (breakout at R4/S4 for continuation, fade at R3/S3) capture 
high-probability trend moves. The 12h Camarilla provides mathematically derived support/resistance 
levels that work in both bull and bear markets. Volume confirms breakout strength. ATR trailing 
stop (2.5x) manages risk. Discrete sizing (0.25) minimizes fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5739_6h_donchian20_12h_camarilla_vol_v1"
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
        # Calculate Camarilla levels from previous 12h bar (completed bar only)
        h_12h = df_12h['high'].values
        l_12h = df_12h['low'].values
        c_12h = df_12h['close'].values
        
        # Camarilla levels: based on previous bar's range
        # R4 = C + ((H-L) * 1.1/2)
        # R3 = C + ((H-L) * 1.1/4)
        # S3 = C - ((H-L) * 1.1/4)
        # S4 = C - ((H-L) * 1.1/2)
        camarilla_r4 = c_12h + ((h_12h - l_12h) * 1.1 / 2)
        camarilla_r3 = c_12h + ((h_12h - l_12h) * 1.1 / 4)
        camarilla_s3 = c_12h - ((h_12h - l_12h) * 1.1 / 4)
        camarilla_s4 = c_12h - ((h_12h - l_12h) * 1.1 / 2)
    else:
        camarilla_r4 = camarilla_r3 = camarilla_s3 = camarilla_s4 = np.array([])
    
    # Align 12h Camarilla levels to 6h timeframe (shifted by 1 for completed 12h bars only)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
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
    
    warmup = max(20, 20, 14)  # Donchian, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below S3 (mean reversion zone)
                if price <= stop_price or price <= camarilla_s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above R3 (mean reversion zone)
                if price >= stop_price or price >= camarilla_r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Camarilla logic: 
        # - Breakout at R4/S4 = continuation signal
        # - Fade at R3/S3 = mean reversion (avoid)
        long_breakout = breakout_up and price >= camarilla_r4_aligned[i-1]
        short_breakout = breakout_down and price <= camarilla_s4_aligned[i-1]
        # Avoid mean reversion zones
        avoid_long = price <= camarilla_r3_aligned[i] and price >= camarilla_s3_aligned[i]
        avoid_short = price >= camarilla_r3_aligned[i] and price <= camarilla_s3_aligned[i]
        
        # Entry conditions: breakout at R4/S4 with volume confirmation, not in mean reversion zone
        long_setup = long_breakout and volume_confirmed and not avoid_long
        short_setup = short_breakout and volume_confirmed and not avoid_short
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals