#!/usr/bin/env python3
"""
Experiment #5739: 6h Donchian(20) breakout + 12h volume confirmation + 1d ADX trend filter
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 1.8x average and aligned 
with 1d ADX > 25 (trending market) capture high-probability trend continuation moves in both 
bull and bear markets. The 1d ADX ensures we only trade in trending regimes, reducing whipsaw 
in ranging markets. Volume confirms breakout strength. ATR trailing stop (2.0x) manages risk. 
Discrete sizing (0.25) minimizes fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5739_6h_donchian20_12h_vol_1d_adx_v1"
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
    
    # === HTF: 12h data for volume confirmation ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        avg_vol_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    else:
        avg_vol_12h = np.full(len(df_12h), np.nan)
    
    # Align 12h average volume to 6h timeframe (shifted by 1 for completed 12h bars only)
    avg_vol_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_vol_12h)
    
    # === HTF: 1d data for ADX trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
        tr_1d[0] = tr1[0]
        atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
        
        # Directional Movement
        up_move = np.diff(high_1d, prepend=high_1d[0])
        down_move = -np.diff(low_1d, prepend=low_1d[0])
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed DM and ATR
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
        atr_1d_smooth = pd.Series(atr_1d).ewm(alpha=1/14, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / np.where(atr_1d_smooth > 0, atr_1d_smooth, 1)
        minus_di = 100 * minus_dm_smooth / np.where(atr_1d_smooth > 0, atr_1d_smooth, 1)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
        adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    else:
        adx_1d = np.full(len(df_1d), np.nan)
    
    # Align 1d ADX to 6h timeframe (shifted by 1 for completed 1d bars only)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = max(20, 20, 14)  # Donchian, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_vol_12h_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR ADX < 20 (trend weakening)
                if price <= stop_price or adx_1d_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR ADX < 20 (trend weakening)
                if price >= stop_price or adx_1d_aligned[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume[i] > 1.8 * avg_vol_12h_aligned[i]
        
        # 1d ADX trend filter: only trade when ADX > 25 (trending market)
        trending_market = adx_1d_aligned[i] > 25
        
        # Entry conditions: breakout in trending market with volume confirmation
        long_setup = breakout_up and volume_confirmed and trending_market
        short_setup = breakout_down and volume_confirmed and trending_market
        
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