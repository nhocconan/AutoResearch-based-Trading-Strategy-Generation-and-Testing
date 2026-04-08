#!/usr/bin/env python3
"""
Experiment #5597: 4h Donchian(20) breakout + daily pivot direction + volume confirmation
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 1.8x average and aligned 
with daily pivot (using actual 1d data) capture high-probability moves. Daily pivot 
provides structural support/resistance from higher timeframe, reducing false breakouts. 
ATR-based trailing stop (2.0x ATR) limits drawdown. Discrete position sizing (0.25) 
minimizes fee churn. Works in bull (breakouts with daily pivot support) and bear 
(breakouts with daily pivot resistance). Target: 19-50 trades/year (75-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5597_4h_donchian20_1d_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for Daily Pivot ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate daily pivot from prior day's OHLC
        daily_high = pd.Series(df_1d['high'].values).shift(1)  # prior day's high
        daily_low = pd.Series(df_1d['low'].values).shift(1)    # prior day's low
        daily_close = pd.Series(df_1d['close'].values).shift(1) # prior day's close
        
        # Daily Pivot Point (PP) = (H + L + C) / 3
        pp = (daily_high + daily_low + daily_close) / 3.0
        # Daily R3 = PP + 2*(H - L)
        r3 = pp + 2.0 * (daily_high - daily_low)
        # Daily S3 = PP - 2*(H - L)
        s3 = pp - 2.0 * (daily_high - daily_low)
        # Daily R4 = PP + 3*(H - L)  (breakout level)
        r4 = pp + 3.0 * (daily_high - daily_low)
        # Daily S4 = PP - 3*(H - L)  (breakdown level)
        s4 = pp - 3.0 * (daily_high - daily_low)
        
        # Align to LTF (4h)
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    else:
        pp_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
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
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR price breaks below S3 (mean reversion) OR breaks below S4 (acceleration)
                if price <= stop_price or price <= s3_aligned[i] or price <= s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR price breaks above R3 (mean reversion) OR breaks above R4 (acceleration)
                if price >= stop_price or price >= r3_aligned[i] or price >= r4_aligned[i]:
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
        
        # Determine bias from Daily Pivot levels
        # In ranging markets: fade at R3/S3 (mean reversion)
        # In trending markets: breakout continuation at R4/S4
        long_setup = (breakout_up and volume_confirmed and 
                     (price > r4_aligned[i] or (price > s3_aligned[i] and price < r3_aligned[i])))
        short_setup = (breakout_down and volume_confirmed and 
                      (price < s4_aligned[i] or (price < r3_aligned[i] and price > s3_aligned[i])))
        
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