#!/usr/bin/env python3
"""
Experiment #3824: 1d Donchian(20) breakout + 1w volume confirmation + chop regime filter
HYPOTHESIS: 1d Donchian breakouts capture multi-week swings with 1d volume (>1.3x) confirming participation. 1w Chop > 61.8 filters range markets to avoid false breakouts. Works in bull (breakouts above resistance) and bear (breakdowns below support). Discrete sizing (0.25) minimizes fee drag. Target: 30-80 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3824_1d_donchian20_1w_vol_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Chop regime (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: Choppiness Index(14) for regime filter ===
    def true_range(high, low, prev_close):
        return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    
    prev_close_1w = np.roll(close_1w, 1)
    prev_close_1w[0] = close_1w[0]
    tr_1w = true_range(high_1w, low_1w, prev_close_1w)
    atr_14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    chop_1w = np.full(len(close_1w), np.nan)
    sum_tr_14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    denominator_1w = atr_14_1w * 14
    mask_1w = (denominator_1w != 0) & ~np.isnan(denominator_1w) & ~np.isnan(sum_tr_14_1w)
    chop_1w[mask_1w] = 100 * np.log10(sum_tr_14_1w[mask_1w] / denominator_1w[mask_1w]) / np.log10(14)
    
    # Align 1w Chop to 1d timeframe (shifted by 1 for completed 1w bar)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # === 1d Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
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
            np.isnan(chop_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                # Calculate 1d ATR(14) for stoploss
                tr_1d = true_range(high[:i+1], low[:i+1], np.roll(close[:i+1], 1))
                tr_1d[0] = high[0] - low[0]
                atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().iloc[-1]
                if price < highest_since_entry - 2.0 * atr_14_1d:
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
                tr_1d = true_range(high[:i+1], low[:i+1], np.roll(close[:i+1], 1))
                tr_1d[0] = high[0] - low[0]
                atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().iloc[-1]
                if price > lowest_since_entry + 2.0 * atr_14_1d:
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
        # Require volume spike (> 1.3x average) AND chop > 61.8 (range regime)
        volume_spike = vol_ratio[i] > 1.3
        chop_filter = chop_1w_aligned[i] > 61.8
        
        if volume_spike and chop_filter:
            # Long entry: Price breaks above Donchian upper band
            if price > highest_high[i-1]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band
            elif price < lowest_low[i-1]:
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