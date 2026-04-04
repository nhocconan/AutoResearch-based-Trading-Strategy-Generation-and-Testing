#!/usr/bin/env python3
"""
Experiment #6355: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: Weekly pivot (from 1w) determines institutional bias. On 6h timeframe, 
Donchian(20) breakouts with volume confirmation (>1.5x avg) trade in the direction 
of weekly pivot. In ranging markets (price between weekly R1-S1), fade at weekly 
R2/S2 with volume divergence. Uses discrete sizing (0.25) and ATR trailing stop 
(2.5x) to control drawdown. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6355_6h_donchian20_1w_pivot_vol_v1"
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
    
    # === HTF: 1w data for weekly pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 2:
        # Weekly pivot calculation from prior week's OHLC
        # Pivot = (H + L + C) / 3
        # R1 = 2*P - L
        # S1 = 2*P - H
        # R2 = P + (H - L)
        # S2 = P - (H - L)
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        r1 = 2 * pivot - weekly_low
        s1 = 2 * pivot - weekly_high
        r2 = pivot + (weekly_high - weekly_low)
        s2 = pivot - (weekly_high - weekly_low)
        
        # Align to 6h timeframe (shift by 1 week for prior week's levels)
        pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
        r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
        r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
        s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    else:
        pivot_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
    
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
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i])):
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
                # 3. Price crosses below S1 (mean reversion in range)
                if price <= stop_price or price <= donchian_low[i] or price < s1_aligned[i]:
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
                # 3. Price crosses above R1 (mean reversion in range)
                if price >= stop_price or price >= donchian_high[i] or price > r1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5  # Volume filter
        
        # Determine market regime based on weekly pivot zones:
        # TRENDING: price > R1 or < S1 -> trade breakouts in pivot direction
        # RANGING: price between S1 and R1 -> fade at R2/S2 with volume divergence
        
        long_entry = False
        short_entry = False
        
        if price > r1_aligned[i]:  # Above weekly R1 -> bullish bias
            if breakout_up and volume_confirmed:
                long_entry = True
        elif price < s1_aligned[i]:  # Below weekly S1 -> bearish bias
            if breakout_down and volume_confirmed:
                short_entry = True
        else:  # Between S1 and R1 -> ranging market
            # Fade at R2/S2 with volume confirmation
            if price >= r2_aligned[i] and volume_confirmed:
                short_entry = True  # Fade resistance
            elif price <= s2_aligned[i] and volume_confirmed:
                long_entry = True   # Fade support
        
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