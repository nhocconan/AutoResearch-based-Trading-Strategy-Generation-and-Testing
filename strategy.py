#!/usr/bin/env python3
"""
Experiment #5519: 6h Donchian(20) breakout + 12h ADX trend filter + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 1.5x average and aligned with 
12h ADX > 25 (strong trend) capture sustained momentum moves while avoiding choppy markets. 
The 12h ADX provides a higher timeframe trend strength filter, reducing false breakouts in 
both bull and bear markets. Discrete position sizing (0.25) and ATR-based stoploss (2.0x ATR) 
control risk. Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5519_6h_donchian20_12h_adx_vol_v1"
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
        # Calculate ADX(14) on 12h data
        # True Range
        tr1 = df_12h['high'] - df_12h['low']
        tr2 = np.abs(df_12h['high'] - np.roll(df_12h['close'], 1))
        tr3 = np.abs(df_12h['low'] - np.roll(df_12h['close'], 1))
        tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
        tr_12h[0] = tr1[0]
        atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
        
        # Directional Movement
        up_move = df_12h['high'] - np.roll(df_12h['high'], 1)
        down_move = np.roll(df_12h['low'], 1) - df_12h['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed DM and TR
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
        tr_smooth = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / np.where(tr_smooth > 0, tr_smooth, 1)
        minus_di = 100 * minus_dm_smooth / np.where(tr_smooth > 0, tr_smooth, 1)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
        adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
        
        # Align to LTF (6h) with shift(1) for completed bars only
        adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
        # Strong trend: ADX > 25
        strong_trend = adx_12h_aligned > 25
    else:
        adx_12h_aligned = np.full(n, np.nan)
        strong_trend = np.full(n, False)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
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
    
    warmup = max(20, 20, 20, 14, 14)  # Donchian, volume avg, ATR warmup, ADX lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        # Trade during major sessions: 00-06 UTC (Asia), 07-12 UTC (Europe), 13-20 UTC (US)
        # Avoid 21-23 UTC (low liquidity between sessions)
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
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.0 * ATR below highest since entry
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Weakening trend: ADX < 20
                if price <= stop_price or price <= donchian_low[i] or adx_12h_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.0 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Weakening trend: ADX < 20
                if price >= stop_price or price >= donchian_high[i] or adx_12h_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Entry conditions: breakout + volume + strong trend (ADX > 25)
        if breakout_up and volume_confirmed and strong_trend[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and strong_trend[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals