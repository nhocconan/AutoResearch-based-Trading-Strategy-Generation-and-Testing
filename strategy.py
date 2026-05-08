#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ChoppinessIndex_Regime_ADX_Filter"
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
    
    # Choppiness Index (4h)
    def choppiness_index(high, low, close, window=14):
        atr = np.abs(high - low)
        tr1 = np.abs(high - np.roll(close, 1))
        tr2 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(atr, np.maximum(tr1, tr2))
        tr[0] = atr[0]
        sum_tr = np.nansum(tr)
        
        if np.isnan(sum_tr) or sum_tr == 0:
            return np.full_like(close, 50.0)
            
        highest_high = np.maximum.accumulate(high)
        lowest_low = np.minimum.accumulate(low)
        range_hl = highest_high - lowest_low
        
        chop = 100 * np.log10(sum_tr / range_hl) / np.log10(window)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # ADX (4h)
    def adx(high, low, close, window=14):
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        plus_di = 100 * pd.Series(plus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        return adx
    
    adx_vals = adx(high, low, close, 14)
    
    # Price channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(chop[i]) or np.isnan(adx_vals[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Range market: mean reversion at extremes
            if chop[i] > 61.8 and adx_vals[i] < 25:
                if close[i] <= lowest_low_20[i] and volume[i] > vol_ma20[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= highest_high_20[i] and volume[i] > vol_ma20[i]:
                    signals[i] = -0.25
                    position = -1
            # Trending market: breakout in direction of trend
            elif chop[i] < 38.2 and adx_vals[i] > 25:
                if close[i] > highest_high_20[i] and volume[i] > vol_ma20[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low_20[i] and volume[i] > vol_ma20[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: re-entry of range or opposite breakout
            if (chop[i] > 61.8 and close[i] >= lowest_low_20[i]) or \
               (chop[i] < 38.2 and close[i] < highest_high_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: re-entry of range or opposite breakout
            if (chop[i] > 61.8 and close[i] <= highest_high_20[i]) or \
               (chop[i] < 38.2 and close[i] > lowest_low_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals