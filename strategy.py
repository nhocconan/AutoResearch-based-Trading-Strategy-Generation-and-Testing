#!/usr/bin/env python3
"""
Experiment #5249: 4h Donchian(20) Breakout + 1d Weekly Pivot + Volume Spike
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts aligned with weekly pivot bias (price > weekly pivot = long bias, price < weekly pivot = short bias) capture institutional flow with volume confirmation (>2.0x average volume). Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts. Designed for 19-50 trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag. Works in bull markets (breakouts above weekly pivot with volume) and bear markets (breakdowns below weekly pivot with volume). Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5249_4h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot (using prior week's OHLC) ===
    if len(df_1d) >= 5:
        # Weekly OHLC from daily data (using prior completed week)
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(1).values
        
        # Weekly Pivot Point calculation
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation (2.0x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Weekly pivot bias: price > pivot = long bias, price < pivot = short bias
        pivot_bias_long = price > weekly_pivot_aligned[i]
        pivot_bias_short = price < weekly_pivot_aligned[i]
        
        # Donchian breakout conditions with weekly pivot bias filter
        # Long: Donchian breakout above + price > weekly pivot (bullish bias)
        # Short: Donchian breakdown below + price < weekly pivot (bearish bias)
        breakout_long = (price >= high_roll[i]) and pivot_bias_long and vol_confirm
        breakout_short = (price <= low_roll[i]) and pivot_bias_short and vol_confirm
        
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