#!/usr/bin/env python3
"""
Experiment #6355: 6h Donchian(20) breakout + 1d Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with volume confirmation (>1.8x avg) and Camarilla pivot bias from 1d timeframe capture institutional momentum. 
Camarilla levels (R3/S3 for fade, R4/S4 for breakout) provide mathematically derived support/resistance that adapts to volatility. 
In ranging markets (price between R3-S3), fade extremes; in breakout markets (price >R4 or <S4), continue the breakout direction. 
Uses discrete sizing (0.25) to minimize fee churn. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6355_6h_donchian20_1d_camarilla_vol_v1"
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
        # Camarilla pivot calculation from prior day's OHLC
        # Pivot = (H + L + C) / 3
        # R4 = C + ((H-L) * 1.1/2)
        # R3 = C + ((H-L) * 1.1/4)
        # S3 = C - ((H-L) * 1.1/4)
        # S4 = C - ((H-L) * 1.1/2)
        daily_high = df_1d['high'].values
        daily_low = df_1d['low'].values
        daily_close = df_1d['close'].values
        
        pivot = (daily_high + daily_low + daily_close) / 3.0
        r4 = daily_close + ((daily_high - daily_low) * 1.1 / 2.0)
        r3 = daily_close + ((daily_high - daily_low) * 1.1 / 4.0)
        s3 = daily_close - ((daily_high - daily_low) * 1.1 / 4.0)
        s4 = daily_close - ((daily_high - daily_low) * 1.1 / 2.0)
        
        # Align to 6h timeframe (shift by 1 day for prior day's levels)
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
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
    
    warmup = max(20, 20, 14) + 1  # Donchian, volume avg, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
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
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                # 3. Price crosses below S3 (mean reversion in range)
                if price <= stop_price or price <= donchian_low[i] or price < s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                # 3. Price crosses above R3 (mean reversion in range)
                if price >= stop_price or price >= donchian_high[i] or price > r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8  # Volume filter
        
        # Entry logic based on Camarilla zones:
        # ZONE 1: Breakout continuation (price > R4 or < S4) - trade in breakout direction
        # ZONE 2: Fade extreme (price > R3 or < S3 but within R4/S4) - trade mean reversion
        # ZONE 3: Range (between S3 and R3) - no trade (chop)
        
        long_entry = False
        short_entry = False
        
        if breakout_up and volume_confirmed:
            # Long breakout continuation above R4
            if price > r4_aligned[i]:
                long_entry = True
            # Long fade from S3 (mean reversion) - only if not already above R4
            elif price > s3_aligned[i] and price <= r4_aligned[i]:
                long_entry = True
                
        if breakout_down and volume_confirmed:
            # Short breakdown continuation below S4
            if price < s4_aligned[i]:
                short_entry = True
            # Short fade from R3 (mean reversion) - only if not already below S4
            elif price < r3_aligned[i] and price >= s4_aligned[i]:
                short_entry = True
        
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