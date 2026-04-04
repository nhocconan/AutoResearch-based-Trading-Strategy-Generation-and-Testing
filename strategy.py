#!/usr/bin/env python3
"""
Experiment #5622: 12h Donchian(20) breakout + 1d ADX trend + volume confirmation
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts with volume > 1.8x average and aligned 
with 1d ADX(14) > 25 (strong trend) capture high-probability trend continuation moves. 
The 1d ADX filter ensures we only trade in strong trending regimes, reducing whipsaws 
in choppy markets. Volume confirmation validates breakout strength. Works in both bull 
and bear markets by trading breakouts in the direction of the 1d trend. 
ATR-based trailing stop (2.0x ATR) manages risk. Discrete position sizing (0.25) 
minimizes fee churn. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5622_12h_donchian20_1d_adx_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for ADX(14) trend strength ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        # Calculate ADX(14) on 1d data
        high_1d = pd.Series(df_1d['high'].values)
        low_1d = pd.Series(df_1d['low'].values)
        close_1d = pd.Series(df_1d['close'].values)
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - close_1d.shift(1))
        tr3 = np.abs(low_1d - close_1d.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_1d = tr.rolling(window=14, min_periods=14).mean()
        
        # Directional Movement
        up_move = high_1d - high_1d.shift(1)
        down_move = low_1d.shift(1) - low_1d
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_1d)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_1d)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14, min_periods=14).mean()
        
        adx_values = adx.values
    else:
        adx_values = np.full(len(df_1d), np.nan)
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 12h Indicators: ATR(14) for trailing stop ===
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
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR ADX weakens (< 20) OR price breaks below Donchian low
                if price <= stop_price or adx_1d_aligned[i] < 20 or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR ADX weakens (< 20) OR price breaks above Donchian high
                if price >= stop_price or adx_1d_aligned[i] < 20 or price >= donchian_high[i]:
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
        strong_trend = adx_1d_aligned[i] > 25
        
        # Trend filter: breakout in direction of strong 1d trend
        # Long: breakout above Donchian high with strong uptrend (ADX > 25)
        # Short: breakout below Donchian low with strong downtrend (ADX > 25)
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