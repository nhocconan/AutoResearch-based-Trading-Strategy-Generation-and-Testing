#!/usr/bin/env python3
"""
Experiment #5265: 12h Donchian(20) breakout + volume spike + 1d EMA50 regime filter
HYPOTHESIS: On 12h timeframe, price breaking Donchian(20) channels from 1d timeframe with volume spike (>1.8x) in the direction of 1d trend (price > 1d EMA50 = bullish, < 1d EMA50 = bearish) captures institutional breakouts while avoiding false moves. Uses discrete position sizing (0.25) and ATR trailing stop (2.0x) to manage risk. Designed for 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize fee drag. Works in bull markets (breakouts continue uptrend) and bear markets (breakouts continue downtrend) by aligning with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5265_12h_donchian_breakout_vol_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 1d data for Donchian channels (structure) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        # Donchian(20) on prior completed 1d bar (shift(1) in align)
        donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
        donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
        donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
        donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    else:
        donch_high_aligned = np.full(n, np.nan)
        donch_low_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for regime filter (EMA50) ===
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Volume confirmation (1.8x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50)  # Donchian, Vol MA, ATR, EMA warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.8x)
        vol_confirm = vol_ratio[i] > 1.8
        
        # Regime filter: bullish if price > 1d EMA50, bearish if price < 1d EMA50
        regime_bullish = price > ema_50_aligned[i]
        regime_bearish = price < ema_50_aligned[i]
        
        # Donchian breakout in regime direction
        breakout_long = (price >= donch_high_aligned[i]) and regime_bullish and vol_confirm
        breakout_short = (price <= donch_low_aligned[i]) and regime_bearish and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals