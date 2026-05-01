#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot filter and volume confirmation.
# Long when price breaks above 6h Donchian upper AND weekly pivot is bullish (price > weekly VWAP) AND volume > 1.5x 6h volume median.
# Short when price breaks below 6h Donchian lower AND weekly pivot is bearish (price < weekly VWAP) AND volume > 1.5x 6h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian provides trend structure, weekly VWAP filters regime (bull/bear/range), volume confirms momentum.
# Target: 12-30 trades/year on 6h timeframe (~50-120 total over 4 years).
# Novelty: Weekly VWAP as regime filter is underutilized; combines intraday breakout with higher-timeframe institutional reference.

name = "6h_Donchian20_WeeklyVWAP_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h volume median (20-period for stability)
    vol_median_6h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 6h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly VWAP (typical price * volume) / volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    tp_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    vol_1w = df_1w['volume'].values
    vwap_1w = (tp_1w * vol_1w).cumsum() / vol_1w.cumsum()
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, volume, and weekly VWAP
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(vol_median_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Weekly pivot filter: bullish if price > weekly VWAP, bearish if price < weekly VWAP
        weekly_bullish = curr_close > vwap_1w_aligned[i]
        weekly_bearish = curr_close < vwap_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 6h volume median
        if vol_median_6h[i] <= 0 or np.isnan(vol_median_6h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_6h[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper AND weekly bullish AND volume spike
            if curr_high > highest_20[i] and weekly_bullish and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower AND weekly bearish AND volume spike
            elif curr_low < lowest_20[i] and weekly_bearish and volume_confirm:
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
            # Exit: price breaks below Donchian lower OR weekly turns bearish
            elif curr_low < lowest_20[i] or not weekly_bullish:
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
            # Exit: price breaks above Donchian upper OR weekly turns bullish
            elif curr_high > highest_20[i] or not weekly_bearish:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals