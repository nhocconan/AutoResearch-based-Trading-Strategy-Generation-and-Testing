#!/usr/bin/env python3
"""
Experiment #5187: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (from 1d data) capture institutional momentum with controlled overtrading. Weekly pivot provides structural bias (R3/S3 for continuation, R4/S4 for reversal) while volume > 1.5x average confirms participation. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. Works in bull markets (breakouts with bullish bias) and bear markets (breakdowns with bearish bias). Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5187_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot and trend
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot Points (using prior week's H/L/C) ===
    if len(df_1d) >= 5:
        # Calculate weekly OHLC from daily data (simplified: use prior 5 days)
        df_1d_idx = pd.RangeIndex(len(df_1d))
        weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().shift(1).values
        weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().shift(1).values
        
        pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        r1 = 2 * pivot - weekly_low
        s1 = 2 * pivot - weekly_high
        r2 = pivot + (weekly_high - weekly_low)
        s2 = pivot - (weekly_high - weekly_low)
        r3 = weekly_high + 2 * (pivot - weekly_low)
        s3 = weekly_low - 2 * (weekly_high - pivot)
        r4 = weekly_high + 3 * (pivot - weekly_low)
        s4 = weekly_low - 3 * (weekly_high - pivot)
        
        # Align to 6h timeframe
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
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
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Determine pivot bias: price > pivot = bullish, price < pivot = bearish
        bullish_bias = price > pivot_aligned[i]
        bearish_bias = price < pivot_aligned[i]
        
        # Donchian breakout conditions with pivot bias and volume confirmation
        # Long: Donchian breakout above + bullish bias + vol confirm
        # Short: Donchian breakdown below + bearish bias + vol confirm
        # Special handling for extreme levels (R4/S4) as continuation signals
        breakout_long = (
            (price >= high_roll[i]) and 
            bullish_bias and 
            vol_confirm and
            ((price <= r4_aligned[i]) or (price > r4_aligned[i]))  # Allow breakout at/beyond R4
        )
        breakout_short = (
            (price <= low_roll[i]) and 
            bearish_bias and 
            vol_confirm and
            ((price >= s4_aligned[i]) or (price < s4_aligned[i]))  # Allow breakdown at/beyond S4
        )
        
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