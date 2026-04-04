#!/usr/bin/env python3
"""
Experiment #4007: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot bias (price above/below weekly pivot) capture high-probability trends. Volume > 1.6x MA(20) confirms participation. Weekly pivot filter reduces false breakouts in ranging markets. Discrete sizing (0.25) and ATR(20) trailing stop (2.0x) control risk. Target: 75-150 trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4007_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Weekly pivot from prior week's OHLC (requires 5 days)
        # We'll calculate it manually since we have daily data
        # For each 6h bar, we need the prior week's H/L/C
        # Simpler: use prior week's close as trend filter
        weekly_close = df_1d['close'].values
        # Align weekly close to 6h timeframe (shifted by 1 week for no look-ahead)
        weekly_close_aligned = align_htf_to_ltf(prices, df_1d, weekly_close)
        # Further align to prior week by shifting another 5 days (approx 1 week)
        # Since 1d to 6h is 4x, 5 days = 20 periods
        weekly_close_aligned = np.roll(weekly_close_aligned, 20)
        weekly_close_aligned[:20] = np.nan
    else:
        weekly_close_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(20) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10, 20 + 10, 20 + 5)  # DC lookback, vol MA, ATR buffer, weekly shift
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_close_aligned[i])):
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
        # Require volume spike (> 1.6x average) to filter noise
        volume_spike = vol_ratio[i] > 1.6
        
        if volume_spike:
            # Determine trend alignment from weekly close (prior week)
            bullish_alignment = price > weekly_close_aligned[i]
            bearish_alignment = price < weekly_close_aligned[i]
            
            # Breakout conditions using Donchian
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Long: bullish alignment + breakout above Donchian high
            if bullish_alignment and breakout_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: bearish alignment + breakout below Donchian low
            elif bearish_alignment and breakout_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals