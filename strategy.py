#!/usr/bin/env python3
"""
Experiment #2575: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: Combining 6h Donchian breakouts with weekly pivot bias and volume confirmation
captures institutional participation during trend acceleration. Weekly pivot provides
longer-term structure (bull/bear bias from 1w timeframe), while 6h Donchian breakouts
with volume spikes capture entry timing. Works in both bull (breakouts with volume in uptrend)
and bear (breakdowns with volume in downtrend) markets. Discrete position sizing (0.25)
limits fee drag and targets 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2575_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot points (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate weekly pivot from daily OHLC (using prior week's data)
    # We'll approximate weekly pivot using rolling window on daily data
    # For true weekly pivot, we need weekly OHLC - but we can approximate
    # using the last 5 daily bars (1 week) to calculate pivot
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate rolling weekly pivot (using last 5 daily bars = 1 week)
    # Weekly Pivot = (Weekly High + Weekly Low + Weekly Close) / 3
    # We approximate weekly high/low/close using rolling window on daily data
    roll_high_5 = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    roll_low_5 = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    roll_close_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point
    weekly_pivot = (roll_high_5 + roll_low_5 + roll_close_5) / 3.0
    
    # Weekly support/resistance levels (simplified)
    # R1 = 2*P - Low, S1 = 2*P - High
    weekly_r1 = 2 * weekly_pivot - roll_low_5
    weekly_s1 = 2 * weekly_pivot - roll_high_5
    
    # Bias: above pivot = bullish, below = bearish
    weekly_bias = np.where(close_1d > weekly_pivot, 1, -1)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    
    # === 6h Indicators: Donchian(20) channels, Volume MA(20) ===
    # Donchian channels (20-period high/low)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_bias_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry (using Donchian width as ATR proxy)
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15  # approximate ATR from channel width
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian low (mean reversion)
                elif price < lowest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian high (mean reversion)
                elif price > highest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly pivot bias for longer-term trend filter
        trend_bias = weekly_bias_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and trend_bias != 0:
            # Long entry: price breaks above Donchian high with weekly bullish bias
            if trend_bias > 0 and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with weekly bearish bias
            elif trend_bias < 0 and price < lowest_20[i]:
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

}