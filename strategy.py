#!/usr/bin/env python3
"""
Experiment #4607: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot bias (price > weekly pivot = bullish, < weekly pivot = bearish) and volume spikes (>1.5x average) capture medium-term momentum with lower trade frequency. Weekly pivot provides structural bias from higher timeframe that adapts to changing market regimes. Uses ATR trailing stop (2.0x) for risk management. Discrete sizing (0.25) minimizes fee churn. Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4607_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot calculation (using last 5 days)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot from prior week's daily OHLC (H+L+C)/3
    # We need at least 5 days to form a week
    if len(df_1d) >= 5:
        # Use rolling window of 5 days to get weekly high, low, close
        dh = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
        dl = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
        dc = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
        weekly_pivot = (dh + dl + dc) / 3.0
    else:
        weekly_pivot = np.array([])
    
    # Align weekly pivot to 6h timeframe
    if len(weekly_pivot) > 0:
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
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
    
    warmup = max(20, 20, 14, 5)  # Donchian, vol MA, ATR, weekly pivot warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i])):
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
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        # Higher timeframe trend filter: bullish when price > weekly pivot, bearish when price < weekly pivot
        htf_bullish = price > weekly_pivot_aligned[i]
        htf_bearish = price < weekly_pivot_aligned[i]
        
        # Donchian breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Long conditions: upward breakout above weekly pivot + volume confirmation
        long_entry = breakout_up and htf_bullish and volume_confirm
        
        # Short conditions: downward breakout below weekly pivot + volume confirmation
        short_entry = breakout_down and htf_bearish and volume_confirm
        
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