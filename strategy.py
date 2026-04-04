#!/usr/bin/env python3
"""
Experiment #6167: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R4/S4 for continuation, R3/S3 for mean reversion) capture institutional flow with volume confirmation. Weekly pivot levels act as dynamic support/resistance where smart money enters/exits. In bull markets, breakouts above R4 continue; in bear markets, breakdowns below S4 continue. Range markets see reversals at R3/S3. Volume >2.0x average confirms participation. ATR trailing stop manages risk. Discrete sizing (0.25) minimizes fee churn. Target: 75-200 trades over 4 years.
Timeframe: 6h. HTF: 1d for weekly pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6167_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # === HTF: 1d data for weekly pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week's OHLC (using Friday's close as weekly close approximation)
        # For simplicity, use prior day's OHLC to compute daily pivots, then weekly context
        # In practice, we'd use actual weekly OHLC, but daily pivots still provide meaningful levels
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        pivot_point = (prev_high + prev_low + prev_close) / 3.0
        r1 = 2 * pivot_point - prev_low
        s1 = 2 * pivot_point - prev_high
        r2 = pivot_point + (prev_high - prev_low)
        s2 = pivot_point - (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot_point - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot_point)
        r4 = prev_high + 3 * (pivot_point - prev_low)
        s4 = prev_low - 3 * (prev_high - pivot_point)
        
        # Align to LTF (6h) with shift(1) for completed daily bars only
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
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
    
    warmup = max(20, 20, 14, 5) + 1  # Donchian, volume avg, ATR, pivot + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
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
        volume_confirmed = volume_ratio[i] > 2.0  # Volume filter for stronger signals
        
        # Weekly pivot context
        price_above_r4 = price > r4_aligned[i]
        price_below_s4 = price < s4_aligned[i]
        price_between_r3_s3 = (price > s3_aligned[i]) and (price < r3_aligned[i])
        price_above_r3 = price > r3_aligned[i]
        price_below_s3 = price < s3_aligned[i]
        
        # Entry logic:
        # Strong continuation: breakout beyond R4/S4 with volume
        # Mean reversion: rejection at R3/S3 with volume
        long_continuation = breakout_up and price_above_r4 and volume_confirmed
        long_reversion = (price <= donchian_low[i-1] + 0.1 * atr[i]) and price_above_s3 and price_below_r3 and volume_confirmed and (close[i-1] < s3_aligned[i-1])
        short_continuation = breakout_down and price_below_s4 and volume_confirmed
        short_reversion = (price >= donchian_high[i-1] - 0.1 * atr[i]) and price_below_r3 and price_above_s3 and volume_confirmed and (close[i-1] > r3_aligned[i-1])
        
        long_entry = long_continuation or long_reversion
        short_entry = short_continuation or short_reversion
        
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

</think>