#!/usr/bin/env python3
"""
Experiment #5567: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 1.8x average and aligned 
with weekly pivot direction (price above weekly pivot = bullish bias, below = bearish bias) 
capture high-probability trend moves. Weekly pivot provides structural support/resistance 
from higher timeframe, reducing false breakouts in ranging markets. ATR-based trailing stop 
limits drawdown. Target: 12-37 trades/year (50-150 total over 4 years) with discrete 
position sizing (0.25) to minimize fee drag. Works in bull (breakouts with pivot support) 
and bear (breakouts with pivot resistance).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5567_6h_donchian20_1w_pivot_vol_v1"
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
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week's OHLC
        # Weekly high/low/close = rolling window of 5 days (approximation for 1d data)
        weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Align to LTF (6h) with shift(1) for completed bars only
        pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        # Neutral values if insufficient data
        pivot_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 5)  # Donchian, volume avg, ATR, weekly pivot warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR Donchian lower band break
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR Donchian upper band break
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
        volume_confirmed = volume_ratio[i] > 1.8
        
        # Determine bias from weekly pivot
        bullish_bias = price > pivot_aligned[i]
        bearish_bias = price < pivot_aligned[i]
        
        # Long: breakout above Donchian high with volume AND bullish bias from weekly pivot
        long_entry = breakout_up and volume_confirmed and bullish_bias
        # Short: breakout below Donchian low with volume AND bearish bias from weekly pivot
        short_entry = breakout_down and volume_confirmed and bearish_bias
        
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