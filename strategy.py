#!/usr/bin/env python3
"""
Experiment #5034: 1h Donchian(20) Breakout + 4h/1d EMA Filter + Volume Spike + Session Filter
HYPOTHESIS: On 1h timeframe, Donchian(20) breakouts aligned with 4h and 1d EMA trend capture momentum with reduced false signals. 
4h EMA(50) and 1d EMA(200) act as regime filters: only long when price > both EMAs, short when price < both EMAs. 
Volume > 1.5x average confirms participation. Session filter (08-20 UTC) reduces off-hour noise. 
ATR(14) trailing stop (2.0x) manages risk. Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5034_1h_donchian20_4h_1d_ema_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours ONCE
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h and 1d data for EMA filters ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: EMA(50) for trend filter ===
    if len(df_4h) >= 50:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: EMA(200) for long-term trend filter ===
    if len(df_1d) >= 200:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 50, 200, 14)  # Donchian, Vol MA, 4h EMA, 1d EMA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        hour = hours[i]
        
        # --- Session Filter: 08-20 UTC ---
        if not (8 <= hour <= 20):
            if in_position:
                # Maintain position during off-hours but don't allow new entries
                if position_side > 0:
                    highest_since_entry = max(highest_since_entry, high[i])
                    if price < highest_since_entry - 2.0 * atr[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:
                    lowest_since_entry = min(lowest_since_entry, low[i])
                    if price > lowest_since_entry + 2.0 * atr[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
            else:
                signals[i] = 0.0
            continue
        
        # --- Exit Logic (during session) ---
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
        
        # --- New Position Entry Logic (only during session) ---
        # Trend filters: price above both EMAs for long, below both for short
        uptrend = price > ema_4h_aligned[i] and price > ema_1d_aligned[i]
        downtrend = price < ema_4h_aligned[i] and price < ema_1d_aligned[i]
        
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        breakout_long = (price >= high_roll[i]) and uptrend and vol_confirm
        breakout_short = (price <= low_roll[i]) and downtrend and vol_confirm
        
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