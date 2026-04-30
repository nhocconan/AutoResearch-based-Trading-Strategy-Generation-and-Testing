#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter
# Donchian channel breakouts capture trending moves. Volume confirmation ensures breakout validity.
# Choppiness index (CHOP) regime filter: CHOP > 61.8 = range (avoid false breakouts), CHOP < 38.2 = trending (favor breakouts).
# Works in bull via upside breakouts, in bear via downside breakouts. Discrete sizing 0.25 minimizes fee churn.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits.

name = "4h_Donchian20_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Choppiness Index regime filter (14-period)
    chop_period = 14
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # first TR
    atr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).mean().values
    highest_high_chop = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_chop = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr.sum() / (highest_high_chop - lowest_low_chop)) / np.log10(chop_period)
    # Fix division by zero and edge cases
    chop = np.where((highest_high_chop - lowest_low_chop) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    chop_market = chop < 38.2  # trending market favors breakouts
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(lookback, 20, chop_period)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_volume_spike = volume_spike[i]
        curr_chop_market = chop_market[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trending market (low chop)
            if curr_volume_spike and curr_chop_market:
                # Bullish breakout: price breaks above Donchian high
                if curr_close > curr_highest_high:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below Donchian low
                elif curr_close < curr_lowest_low:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below Donchian low (stop loss)
            if curr_close < curr_lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high (stop loss)
            if curr_close > curr_highest_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals