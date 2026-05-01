#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d chop regime filter.
# Long when price breaks above Donchian upper band with volume > 1.5x 20-bar average and 1d chop < 38.2 (trending).
# Short when price breaks below Donchian lower band with volume confirmation and 1d chop < 38.2.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Designed to capture strong trends in both bull and bear markets while avoiding choppy, ranging conditions.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

name = "4h_Donchian_20_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for consistency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Chop Index(14) for 1d regime filter: <38.2 = trending (breakout), >61.8 = range (mean revert)
    def true_range(h, l, c):
        h_l = h - l
        h_pc = np.abs(np.subtract(h, np.roll(c, 1)))
        l_pc = np.abs(np.subtract(l, np.roll(c, 1)))
        h_pc[0] = np.abs(h[0] - c[0])
        l_pc[0] = np.abs(l[0] - c[0])
        return np.maximum(h_l, np.maximum(h_pc, l_pc))
    
    # Load 1d data ONCE before loop for chop regime (HTF filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_chop_1d = true_range(high_1d, low_1d, close_1d)
    atr_14_1d = pd.Series(tr_chop_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denom_1d = highest_high_14_1d - lowest_low_14_1d
    # Avoid division by zero
    chop_1d = np.where(denom_1d != 0, 100 * np.log10(atr_14_1d / denom_1d) / np.log10(14), 50.0)
    
    # Align 1d chop to 4h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR and Donchian
    start_idx = 20
    
    for i in range(start_idx, n):
        # Session filter: 4h timeframe less sensitive, but keep structure
        # if not (0 <= hours[i] <= 23):  # always true, kept for structure
        #     signals[i] = 0.0
        #     continue
        
        if (np.isnan(atr[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0 or np.isnan(vol_ma):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.5)
        
        # Donchian breakout conditions
        breakout_up = curr_high > highest_high_20[i]
        breakout_down = curr_low < lowest_low_20[i]
        
        # Chop regime filter: only trade in trending market (chop < 38.2)
        trend_filter = chop_1d_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume confirmation AND trend regime
            if (breakout_up and 
                volume_confirm and 
                trend_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout down AND volume confirmation AND trend regime
            elif (breakout_down and 
                  volume_confirm and 
                  trend_filter):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR trend regime ends (choppy)
            elif (curr_low >= lowest_low_20[i] and curr_low <= highest_high_20[i]) or \
                 chop_1d_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR trend regime ends (choppy)
            elif (curr_high >= lowest_low_20[i] and curr_high <= highest_high_20[i]) or \
                 chop_1d_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals