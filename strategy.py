#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX filter.
# Uses daily ATR for stoploss (2*ATR). Long when price breaks above upper band
# with volume > 20-period average and ADX > 25 (trending market).
# Short when price breaks below lower band with same filters.
# Works in both bull (breakouts continue) and bear (breakdowns continue) markets.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "4h_donchian20_vol_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX calculation (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        # ADX filter: require trending market
        adx_filter = adx[i] > 25
        
        if position == 1:  # long position
            # Stoploss: 2*ATR against position
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price re-enters Donchian channel (mean reversion)
            elif close[i] < high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2*ATR against position
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price re-enters Donchian channel (mean reversion)
            elif close[i] > low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with filters
            if vol_filter and adx_filter:
                # Breakout above upper band
                if close[i] > high_20[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Breakdown below lower band
                elif close[i] < low_20[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals