#!/usr/bin/env python3
"""
Experiment #6187: 6h Donchian(20) breakout + 1d pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d Camarilla pivot levels (R3/S3 for mean reversion, 
R4/S4 for breakout) capture medium-term momentum with institutional volume confirmation. 
In bear markets, fading at R3/S3 works; in bull markets, breakouts at R4/S4 work. 
Volume >1.5x average confirms participation. Discrete sizing (0.25) manages fee drag. 
Target: 75-150 trades over 4 years (19-37/year).
Timeframe: 6h. HTF: 1d for Camarilla pivots.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6187_6h_donchian20_1d_pivot_vol_v1"
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
        # Calculate Camarilla pivots from previous day's OHLC
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_range = prev_high - prev_low
        
        # Camarilla levels: R4 = close + range * 1.1/2, R3 = close + range * 1.1/4, etc.
        camarilla_r4 = prev_close + prev_range * 1.1 / 2
        camarilla_r3 = prev_close + prev_range * 1.1 / 4
        camarilla_s3 = prev_close - prev_range * 1.1 / 4
        camarilla_s4 = prev_close - prev_range * 1.1 / 2
        
        # Align to 6h timeframe (shift(1) inside align_htf_to_ltf for completed bars)
        r4_1d = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        s4_1d = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        r4_1d = r3_1d = s3_1d = s4_1d = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 2) + 1  # Donchian, volume avg, ATR, 1d pivot + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21:00-23:59 UTC) ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r4_1d[i]) or np.isnan(r3_1d[i]) or
            np.isnan(s3_1d[i]) or np.isnan(s4_1d[i])):
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
        volume_confirmed = volume_ratio[i] > 1.5  # Volume filter for stronger signals
        
        # Camarilla pivot logic:
        # In range/choppy markets: fade at R3/S3 (mean reversion)
        # In trending markets: breakout at R4/S4 (continuation)
        # Determine regime by price position relative to R3/S3
        near_r3 = abs(price - r3_1d[i]) < (r4_1d[i] - r3_1d[i]) * 0.1  # Within 10% of R3
        near_s3 = abs(price - s3_1d[i]) < (s3_1d[i] - s4_1d[i]) * 0.1  # Within 10% of S3
        
        # Long conditions:
        # 1. Breakout above Donchian high with volume AND above R4 (bullish breakout)
        # 2. OR bounce from S3 with volume (mean reversion in range)
        long_breakout = breakout_up and volume_confirmed and price > r4_1d[i]
        long_reversion = (price <= s3_1d[i] * 1.005) and volume_confirmed and near_s3 and price > s4_1d[i]
        long_entry = long_breakout or long_reversion
        
        # Short conditions:
        # 1. Breakout below Donchian low with volume AND below S4 (bearish breakout)
        # 2. OR rejection at R3 with volume (mean reversion in range)
        short_breakout = breakout_down and volume_confirmed and price < s4_1d[i]
        short_reversion = (price >= r3_1d[i] * 0.995) and volume_confirmed and near_r3 and price < r4_1d[i]
        short_entry = short_breakout or short_reversion
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals