#!/usr/bin/env python3
"""
Experiment #3783: 4h Donchian(20) breakout + 12h HMA trend + 1d volume confirmation
HYPOTHESIS: 4h Donchian breakouts capture swing moves aligned with 12h HMA trend, with 1d volume spike (>1.5x) confirming institutional participation. Works in bull markets (breakouts above VHN) and bear markets (breakdowns below VHN). Discrete position sizing (0.25) minimizes fee drag. Target: 100-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3783_4h_donchian20_12h_hma_1d_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA(21) trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # Calculate HMA(21) on 12h close
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).ewm(span=half_len, adjust=False).mean().values
    wma_full = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    hma_12h_raw = 2 * wma_half - wma_full
    hma_12h = pd.Series(hma_12h_raw).ewm(span=sqrt_len, adjust=False).mean().values
    # Align to 4h timeframe (shifted by 1 for completed 12h bar)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === HTF: 1d data for volume profile VHN (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume profile high-volume node (VHN) - price level with max volume
    nbins = 50
    vhn_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i < 1:
            continue
        # Create volume histogram for this 1d bar
        hist, bin_edges = np.histogram(
            [high_1d[i], low_1d[i], close_1d[i]],
            bins=nbins,
            range=(low_1d[i], high_1d[i]),
            weights=[volume_1d[i], volume_1d[i], volume_1d[i]]
        )
        if np.sum(hist) > 0:
            max_bin_idx = np.argmax(hist)
            vhn_1d[i] = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    
    # Align 1d VHN to 4h timeframe (shifted by 1 for completed 1d bar)
    vhn_1d_aligned = align_htf_to_ltf(prices, df_1d, vhn_1d)
    
    # === 4h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(vhn_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * (high[i] - low[i]):  # Simplified ATR proxy
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * (high[i] - low[i]):  # Simplified ATR proxy
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above 12h HMA (bullish breakout with trend)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > hma_12h_aligned[i]):   # Above 12h HMA trend
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below 12h HMA (bearish breakdown against trend)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < hma_12h_aligned[i]):   # Below 12h HMA trend
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