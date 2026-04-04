#!/usr/bin/env python3
"""
Experiment #5819: 6h Donchian(20) breakout + 12h Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Camarilla pivot levels (R3/S3 for fade, R4/S4 for continuation) capture institutional order flow with proper frequency. Volume confirmation filters false breakouts. ATR-based trailing stop manages risk. Discrete position sizing (0.25) minimizes fee churn. Targets 75-200 trades over 4 years. Works in bull markets (breakouts with bullish Camarilla bias) and avoids false signals in bear via dual HTF regime filter. Timeframe: 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5819_6h_donchian20_12h_camarilla_vol_v1"
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
        # Calculate Camarilla pivots from previous 12h bar
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # Camarilla pivot formula
        pivot_12h = (high_12h + low_12h + close_12h) / 3.0
        range_12h = high_12h - low_12h
        
        # Resistance levels
        r3_12h = close_12h + range_12h * 1.1 / 4.0
        r4_12h = close_12h + range_12h * 1.1 / 2.0
        # Support levels
        s3_12h = close_12h - range_12h * 1.1 / 4.0
        s4_12h = close_12h - range_12h * 1.1 / 2.0
    else:
        # Not enough data - fill with NaN
        pivot_12h = np.full(len(df_12h), np.nan)
        r3_12h = np.full(len(df_12h), np.nan)
        r4_12h = np.full(len(df_12h), np.nan)
        s3_12h = np.full(len(df_12h), np.nan)
        s4_12h = np.full(len(df_12h), np.nan)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed bars only)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
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
    
    warmup = max(20, 20, 20, 14)  # Donchian, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(s4_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
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
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Camarilla-based regime: 
        # Long bias: price above R3 (strong bullish) or between S3 and R3 (neutral)
        # Short bias: price below S3 (strong bearish) or between S3 and R3 (neutral)
        # Continuation signals: breakout beyond R4/S4
        # Fade signals: rejection at R3/S3
        long_continuation = breakout_up and price > r4_12h_aligned[i-1]
        long_fade = not breakout_up and price < r3_12h_aligned[i-1] and price > s3_12h_aligned[i-1]
        short_continuation = breakout_down and price < s4_12h_aligned[i-1]
        short_fade = not breakout_down and price > s3_12h_aligned[i-1] and price < r3_12h_aligned[i-1]
        
        # Entry conditions: 
        # Long: continuation breakout above R4 OR fade from R3 with volume confirmation
        # Short: continuation breakout below S4 OR fade from S3 with volume confirmation
        long_setup = (long_continuation or long_fade) and volume_confirmed
        short_setup = (short_continuation or short_fade) and volume_confirmed
        
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