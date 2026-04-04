#!/usr/bin/env python3
"""
Experiment #3826: 4h Donchian(20) breakout + 1d volume confirmation + chop regime filter
HYPOTHESIS: 4h Donchian breakouts capture medium-term swings with 1d volume (>1.5x) confirming institutional participation. Choppiness Index (14) > 61.8 filters range markets to avoid false breakouts. Works in bull markets (breakouts above resistance) and bear markets (breakdowns below support). Discrete position sizing (0.25) minimizes fee drag. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3826_4h_donchian20_1d_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
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
    
    # === 4h Indicators: Choppiness Index(14) for regime filter ===
    def true_range(high, low, prev_close):
        return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.full(n, np.nan)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = atr_14 * 14
    mask = (denominator != 0) & ~np.isnan(denominator) & ~np.isnan(sum_tr_14)
    chop[mask] = 100 * np.log10(sum_tr_14[mask] / denominator[mask]) / np.log10(14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vhn_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr_14[i]:
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
                if price > lowest_since_entry + 2.0 * atr_14[i]:
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
        # Require volume spike (> 1.5x average) AND chop > 61.8 (range regime)
        volume_spike = vol_ratio[i] > 1.5
        chop_filter = chop[i] > 61.8
        
        if volume_spike and chop_filter:
            # Long entry: Price breaks above Donchian upper band AND above 1d VHN (bullish breakout with volume confirmation)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > vhn_1d_aligned[i]):    # Above 1d VHN (institutional interest level)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below 1d VHN (bearish breakdown with volume confirmation)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < vhn_1d_aligned[i]):    # Below 1d VHN (institutional interest level)
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