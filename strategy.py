#!/usr/bin/env python3
"""
Experiment #3707: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture intermediate-term momentum with weekly pivot levels providing structural bias from higher timeframe. Volume spike confirms breakout authenticity. This avoids whipsaw in ranging markets and works in both bull (breakouts with trend) and bear (breakouts against trend filtered by weekly pivot) regimes. Targets 75-150 trades over 4 years (19-37/year) with strict 3-condition confluence. Position size 0.25 manages drawdown from 2022 crash while allowing profit accumulation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3707_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(high_1d)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Week pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # R1 = (2 * Pivot) - Prior Week Low
    # S1 = (2 * Pivot) - Prior Week High
    # R2 = Pivot + (Prior Week High - Prior Week Low)
    # S2 = Pivot - (Prior Week High - Prior Week Low)
    # R3 = Prior Week High + 2 * (Pivot - Prior Week Low)
    # S3 = Prior Week Low - 2 * (Prior Week High - Pivot)
    # R4 = Prior Week High + 3 * (Pivot - Prior Week Low)
    # S4 = Prior Week Low - 3 * (Prior Week High - Pivot)
    
    # Use shift(1) to get prior week's data (already handled by align_htf_to_ltf later)
    # But we need to calculate pivot from prior week's OHLC
    # For weekly pivot, we need to group by week first
    # However, since we have daily data, we can approximate weekly pivot using 5-day lookback
    # More accurate: calculate weekly OHLC then apply pivot formula
    
    # Group daily data into weeks (approximate: 5 trading days)
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values  # approximate weekly close
    
    # Calculate pivot points from prior week's OHLC
    week_pivot = (week_high + week_low + week_close) / 3.0
    
    # Calculate support/resistance levels
    r1 = (2 * week_pivot) - week_low
    s1 = (2 * week_pivot) - week_high
    r2 = week_pivot + (week_high - week_low)
    s2 = week_pivot - (week_high - week_low)
    r3 = week_high + 2 * (week_pivot - week_low)
    s3 = week_low - 2 * (week_high - week_pivot)
    r4 = week_high + 3 * (week_pivot - week_low)
    s4 = week_low - 3 * (week_high - week_pivot)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed weekly bar)
    week_pivot_aligned = align_htf_to_ltf(prices, df_1d, week_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20, 14, 5)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(week_pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
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
                # Exit if price breaks below weekly S3 (support breakdown)
                elif price < s3_aligned[i]:
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
                # Exit if price breaks above weekly R3 (resistance breakdown)
                elif price > r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above weekly R3 (bullish breakout)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > r3_aligned[i]):        # Above weekly R3 (bullish bias)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below weekly S3 (bearish breakout)
            elif (price < lowest_low[i-1] and   # Breakout below previous period's low
                  price < s3_aligned[i]):       # Below weekly S3 (bearish bias)
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