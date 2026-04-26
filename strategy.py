#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_RegimeFilter_ATRStop
Hypothesis: Donchian(20) breakouts on 4h with volume confirmation and chop regime filter capture strong trends while avoiding whipsaw in ranging markets. ATR-based stoploss and discrete sizing (0.0, ±0.25) control risk. Targets ~20-30 trades/year to stay within optimal trade frequency for 4h timeframe. Works in both bull (breakouts up) and bear (breakdowns down) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter (using EMA50 on 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_14 = atr  # already calculated ATR(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_ll = highest_high - lowest_low
    chop = 100 * np.log10(sum_atr_14 / np.maximum(hh_ll, 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), ATR(14), volume MA(20), choppy(14)
    start_idx = max(20, 14, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_confirmed = vol_ratio[i] > 2.0  # volume at least 2.0x average
        trend_1d_up = close_val > ema_50_1d_aligned[i]
        trend_1d_down = close_val < ema_50_1d_aligned[i]
        
        # Regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
        # We only take trend-following signals in trending markets (CHOP < 38.2)
        trending_market = chop[i] < 38.2
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND 1d trend up AND volume confirmation AND trending market
            long_signal = (close_val > highest_high[i]) and trend_1d_up and vol_confirmed and trending_market
            
            # Short: price breaks below Donchian lower band AND 1d trend down AND volume confirmation AND trending market
            short_signal = (close_val < lowest_low[i]) and trend_1d_down and vol_confirmed and trending_market
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down OR price hits ATR stoploss OR Donchian breakout in opposite direction
            if (not trend_1d_up) or (close_val < entry_price - 2.0 * atr[i]) or (close_val < lowest_low[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss OR Donchian breakout in opposite direction
            if (not trend_1d_down) or (close_val > entry_price + 2.0 * atr[i]) or (close_val > highest_high[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_RegimeFilter_ATRStop"
timeframe = "4h"
leverage = 1.0