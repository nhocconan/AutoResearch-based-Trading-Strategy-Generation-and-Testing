#!/usr/bin/env python3
"""
Experiment #5539: 6h Donchian(20) breakout + 12h Supertrend + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 1.5x average and aligned with 
12h Supertrend direction capture high-probability trend continuation moves. The 12h Supertrend 
provides robust trend filtering that works across bull/bear regimes, while volume confirmation 
filters false breakouts. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5539_6h_donchian20_12h_supertrend_vol_v1"
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
    
    # === HTF: 12h data for Supertrend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 10:
        # Calculate Supertrend on 12h data
        hl2 = (df_12h['high'] + df_12h['low']) / 2
        # ATR calculation
        tr1 = df_12h['high'] - df_12h['low']
        tr2 = np.abs(df_12h['high'] - df_12h['close'].shift(1))
        tr3 = np.abs(df_12h['low'] - df_12h['close'].shift(1))
        tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_12h = tr_12h.rolling(window=10, min_periods=10).mean()
        # Basic bands
        upper_basic = hl2 + 3.0 * atr_12h
        lower_basic = hl2 - 3.0 * atr_12h
        # Final bands
        upper_band = upper_basic.copy()
        lower_band = lower_basic.copy()
        for i in range(1, len(upper_basic)):
            if close.iloc[i-1] <= upper_band.iloc[i-1]:
                upper_band.iloc[i] = min(upper_basic.iloc[i], upper_band.iloc[i-1])
            else:
                upper_band.iloc[i] = upper_basic.iloc[i]
            if close.iloc[i-1] >= lower_band.iloc[i-1]:
                lower_band.iloc[i] = max(lower_basic.iloc[i], lower_band.iloc[i-1])
            else:
                lower_band.iloc[i] = lower_basic.iloc[i]
        # Supertrend
        supertrend = pd.Series(index=df_12h.index, dtype=float)
        for i in range(len(supertrend)):
            if i == 0:
                supertrend.iloc[i] = upper_band.iloc[i]
            elif supertrend.iloc[i-1] == upper_band.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i] if close.iloc[i] <= upper_band.iloc[i] else lower_band.iloc[i]
            else:
                supertrend.iloc[i] = lower_band.iloc[i] if close.iloc[i] >= lower_band.iloc[i-1] else upper_band.iloc[i]
        # Trend direction: 1 = uptrend (price below Supertrend), -1 = downtrend (price above Supertrend)
        trend_12h = np.where(close.values[:len(supertrend)] <= supertrend.values, 1, -1)
        # Align to LTF (6h) with shift(1) for completed bars only
        trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    else:
        trend_12h_aligned = np.full(n, 0)  # neutral if insufficient data
    
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
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 10, 14)  # Donchian, volume avg, Supertrend ATR, ATR warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR Donchian lower band break OR trend reversal
                if price <= stop_price or price <= donchian_low[i] or (i < len(trend_12h_aligned) and trend_12h_aligned[i] == -1):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR Donchian upper band break OR trend reversal
                if price >= stop_price or price >= donchian_high[i] or (i < len(trend_12h_aligned) and trend_12h_aligned[i] == 1):
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
        
        # Only trade in direction of 12h Supertrend trend
        long_entry = breakout_up and volume_confirmed and (i < len(trend_12h_aligned) and trend_12h_aligned[i] == 1)
        short_entry = breakout_down and volume_confirmed and (i < len(trend_12h_aligned) and trend_12h_aligned[i] == -1)
        
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