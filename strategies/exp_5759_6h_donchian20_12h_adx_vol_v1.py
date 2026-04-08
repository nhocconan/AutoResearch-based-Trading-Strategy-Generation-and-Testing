#!/usr/bin/env python3
"""
Experiment #5759: 6h Donchian(20) breakout + 12h ADX trend filter + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h ADX > 25 capture strong trending moves while avoiding choppy markets. Volume > 1.5x average confirms breakout strength. Uses discrete sizing 0.25 to minimize fees. Designed to work in both bull and bear markets by requiring strong trend (ADX) rather than directional bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5759_6h_donchian20_12h_adx_vol_v1"
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
    
    # === HTF: 12h data for ADX trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 14:
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
        up_move = high_12h - np.roll(high_12h, 1)
        down_move = np.roll(low_12h, 1) - low_12h
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM (Wilder's smoothing)
        def wilder_smooth(x, period):
            result = np.full_like(x, np.nan)
            if len(x) >= period:
                result[period-1] = np.nansum(x[:period])
                for i in range(period, len(x)):
                    result[i] = result[i-1] - (result[i-1] / period) + x[i]
            return result
        
        atr_12h = wilder_smooth(tr_12h, 14)
        plus_dm_12h = wilder_smooth(plus_dm, 14)
        minus_dm_12h = wilder_smooth(minus_dm, 14)
        
        # DI+ and DI-
        plus_di_12h = np.where(atr_12h > 0, (plus_dm_12h / atr_12h) * 100, 0)
        minus_di_12h = np.where(atr_12h > 0, (minus_dm_12h / atr_12h) * 100, 0)
        
        # DX and ADX
        dx_12h = np.where((plus_di_12h + minus_di_12h) > 0, 
                          np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h) * 100, 0)
        adx_12h = wilder_smooth(dx_12h, 14)
    else:
        adx_12h = np.full(len(df_12h), np.nan)
    
    # Align 12h ADX to 6h timeframe (shifted by 1 for completed 12h bars only)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
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
    
    warmup = max(20, 20, 14, 14)  # Donchian, volume avg, ATR, ADX period
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
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
                # Exit: stoploss OR ADX weakens (< 20) OR price breaks below Donchian low
                if price <= stop_price or adx_12h_aligned[i] < 20 or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR ADX weakens (< 20) OR price breaks above Donchian high
                if price >= stop_price or adx_12h_aligned[i] < 20 or price >= donchian_high[i]:
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
        strong_trend = adx_12h_aligned[i] > 25
        
        # Entry conditions: breakout in any direction with volume and strong trend
        long_setup = breakout_up and volume_confirmed and strong_trend
        short_setup = breakout_down and volume_confirmed and strong_trend
        
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