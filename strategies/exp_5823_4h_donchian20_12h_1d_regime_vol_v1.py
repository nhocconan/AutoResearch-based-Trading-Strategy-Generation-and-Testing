#!/usr/bin/env python3
"""
Experiment #5823: 4h Donchian(20) breakout + 12h/1d HTF regime + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with both 12h EMA20 trend and 1d EMA50 trend capture institutional flow with proper frequency. Volume confirmation filters false breakouts. ATR-based trailing stop manages risk. Dual HTF regime filter reduces false signals in bear markets. Discrete position sizing (0.25) minimizes fee churn. Targets 75-200 trades over 4 years. Works in bull markets (breakouts with bullish HTF bias) and avoids false signals in bear via dual HTF regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5823_4h_donchian20_12h_1d_regime_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 12h data for EMA20 trend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        ema_12h = pd.Series(df_12h['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    else:
        ema_12h = np.full(len(df_12h), np.nan)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === HTF: 1d data for EMA50 trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 20, 50, 14)  # Donchian, volume avg, EMA20, EMA50, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
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
        volume_confirmed = volume_ratio[i] > 1.5
        # Dual regime filter: 12h EMA20 trend + 1d EMA50 trend
        regime_long = (price > ema_12h_aligned[i]) and (price > ema_1d_aligned[i])
        regime_short = (price < ema_12h_aligned[i]) and (price < ema_1d_aligned[i])
        
        # Entry conditions: breakout in direction of dual HTF regime with volume confirmation
        long_setup = breakout_up and regime_long and volume_confirmed
        short_setup = breakout_down and regime_short and volume_confirmed
        
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