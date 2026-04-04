#!/usr/bin/env python3
"""
Experiment #5714: 1h Donchian(20) breakout + 4h EMA50 trend + 1d EMA200 filter + volume confirmation
HYPOTHESIS: On 1h timeframe, Donchian(20) breakouts with volume > 1.5x average, aligned with 
4h EMA50 trend (price above EMA50 = bullish, below = bearish), and filtered by 1d EMA200 
(regime: price above EMA200 = bull market bias, below = bear market bias) capture high-probability 
trend continuation moves. The multi-timeframe alignment reduces whipsaw, while volume confirms 
breakout strength. Discrete sizing (0.20) minimizes fee churn. Target: 15-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5714_1h_donchian20_4h_ema50_1d_ema200_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 4h data for EMA50 trend ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 50:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_4h = np.full(len(df_4h), np.nan)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for EMA200 regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 200:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 1h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50, 200)  # Donchian, volume avg, ATR, EMA50, EMA200
    
    for i in range(warmup, n):
        # --- Session Filter: Trade only during active liquidity hours (08-20 UTC) ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR price breaks below 4h EMA50 (trend change) OR 1d EMA200 regime flip
                if price <= stop_price or price <= ema_4h_aligned[i] or price <= ema_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR price breaks above 4h EMA50 (trend change) OR 1d EMA200 regime flip
                if price >= stop_price or price >= ema_4h_aligned[i] or price >= ema_1d_aligned[i]:
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
        
        # 4h EMA50 bias: long above EMA50, short below EMA50
        long_bias_4h = price > ema_4h_aligned[i]
        short_bias_4h = price < ema_4h_aligned[i]
        
        # 1d EMA200 regime filter: long bias above EMA200, short bias below EMA200
        long_bias_1d = price > ema_1d_aligned[i]
        short_bias_1d = price < ema_1d_aligned[i]
        
        # Entry conditions: breakout in direction of both 4h and 1d bias with volume
        long_setup = breakout_up and volume_confirmed and long_bias_4h and long_bias_1d
        short_setup = breakout_down and volume_confirmed and short_bias_4h and short_bias_1d
        
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