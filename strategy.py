#!/usr/bin/env python3
"""
Experiment #5015: 6h Donchian(20) Breakout + 1w Higher High/Lower Low + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly trend (higher highs/lows) capture strong momentum moves with low false breakouts. Volume confirmation (>2x average) ensures institutional participation. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with higher highs) and bear markets (breakdowns with lower lows).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5015_6h_donchian20_1w_hhll_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for HH/LL trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: Higher High/Lower Low (HH/LL) trend ===
    if len(df_1w) >= 2:
        hh = df_1w['high'].rolling(window=2, min_periods=2).max().values
        ll = df_1w['low'].rolling(window=2, min_periods=2).min().values
        # Trend: 1 = HH (bullish), -1 = LL (bearish), 0 = mixed/unclear
        trend_1w = np.where(hh > ll, 1, np.where(hh < ll, -1, 0))
    else:
        trend_1w = np.full(len(df_1w), 0)
    
    # Align HTF HH/LL to 6h timeframe
    if len(trend_1w) > 0:
        trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    else:
        trend_1w_aligned = np.full(n, 0)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (2x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions with HH/LL trend alignment
        breakout_long = (price >= high_roll[i]) and (trend_1w_aligned[i] >= 0) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (trend_1w_aligned[i] <= 0) and vol_confirm
        
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

</think>