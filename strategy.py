#!/usr/bin/env python3
"""
Experiment #6319: 6h Donchian(20) breakout + 12h ADX trend filter + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h timeframe with 12h ADX > 25 as trend filter capture strong momentum moves.
Volume confirmation ensures breakouts have participation. Works in both bull (breakouts with ADX>25 in uptrend) 
and bear (breakouts with ADX>25 in downtrend) markets. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6319_6h_donchian20_12h_adx_vol_v1"
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
    
    # === HTF: 12h data for ADX(14) trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 14:
        # Calculate ADX(14) on 12h data
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # True Range
        tr1 = high_12h - low_12h
        tr2 = np.abs(high_12h - np.roll(close_12h, 1))
        tr3 = np.abs(low_12h - np.roll(close_12h, 1))
        tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
        tr_12h[0] = tr1[0]
        
        # Directional Movement
        dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                           np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
        dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                            np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
        
        # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                result[period-1] = np.mean(data[:period])
                for i in range(period, len(data)):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr_12h = wilder_smooth(tr_12h, 14)
        dm_plus_smooth = wilder_smooth(dm_plus, 14)
        dm_minus_smooth = wilder_smooth(dm_minus, 14)
        
        # DI+ and DI-
        di_plus = np.where(atr_12h > 0, (dm_plus_smooth / atr_12h) * 100, 0)
        di_minus = np.where(atr_12h > 0, (dm_minus_smooth / atr_12h) * 100, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
        adx_12h = wilder_smooth(dx, 14)
        
        # Align to 6h timeframe
        adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    else:
        adx_12h_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 14) + 1  # Donchian, volume avg, ATR, ADX + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(adx_12h_aligned[i])):
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
                # 3. ADX drops below 20 (trend weakening)
                if price <= stop_price or price <= donchian_low[i] or adx_12h_aligned[i] < 20:
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
                # 3. ADX drops below 20 (trend weakening)
                if price >= stop_price or price >= donchian_high[i] or adx_12h_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0  # Strong volume filter
        strong_trend = adx_12h_aligned[i] > 25  # ADX > 25 indicates strong trend
        
        # Entry logic: Donchian breakout with volume AND strong trend (ADX > 25)
        # LONG: breakout above Donchian high + volume + ADX > 25
        # SHORT: breakout below Donchian low + volume + ADX > 25
        long_entry = breakout_up and volume_confirmed and strong_trend
        short_entry = breakout_down and volume_confirmed and strong_trend
        
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