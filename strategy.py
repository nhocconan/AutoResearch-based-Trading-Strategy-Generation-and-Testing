#!/usr/bin/env python3
"""
Experiment #5891: 6h Donchian(20) breakout + 1d Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d Camarilla pivot levels (R3/S3 for fade, R4/S4 for continuation) 
capture institutional order flow. Volume confirmation filters weak breakouts. 
Works in bull markets (breakouts above R4 with volume) and bear markets 
(breakdowns below S4 with volume). Targets 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5891_6h_donchian20_1d_camarilla_vol_v1"
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
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate Camarilla levels for previous day
        camarilla_r4 = np.zeros(len(close_1d))
        camarilla_r3 = np.zeros(len(close_1d))
        camarilla_s3 = np.zeros(len(close_1d))
        camarilla_s4 = np.zeros(len(close_1d))
        
        for i in range(1, len(close_1d)):
            # Previous day's range
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_val = prev_high - prev_low
            
            if range_val > 0:
                camarilla_r4[i] = prev_close + range_val * 1.1 / 2
                camarilla_r3[i] = prev_close + range_val * 1.1 / 4
                camarilla_s3[i] = prev_close - range_val * 1.1 / 4
                camarilla_s4[i] = prev_close - range_val * 1.1 / 2
            else:
                camarilla_r4[i] = camarilla_r3[i] = camarilla_s3[i] = camarilla_s4[i] = prev_close
        
        # Align to 6h timeframe
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
    
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
        
        # Camarilla-based logic:
        # Long: breakout above R4 (continuation) OR bounce from S3 (fade)
        # Short: breakdown below S4 (continuation) OR bounce from R3 (fade)
        long_continuation = breakout_up and price > camarilla_r4_aligned[i-1]
        long_fade = price < camarilla_s3_aligned[i-1] and price > donchian_low[i-1]  # Oversold bounce
        short_continuation = breakout_down and price < camarilla_s4_aligned[i-1]
        short_fade = price > camarilla_r3_aligned[i-1] and price < donchian_high[i-1]  # Overbought bounce
        
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