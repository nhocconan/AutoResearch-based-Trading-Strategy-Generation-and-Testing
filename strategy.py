#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume and ADX trend filter.
# Long when price breaks above upper Donchian channel during strong uptrend (ADX>25) with volume > 1.5x 20-period average.
# Short when price breaks below lower Donchian channel during strong downtrend (ADX>25) with volume confirmation.
# Uses ADX to filter for trending markets only, reducing whipsaws in ranging conditions.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "4h_donchian20_adx_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # ADX calculation (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(np.concatenate([[0], plus_dm])).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(np.concatenate([[0], minus_dm])).ewm(alpha=1/14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if ADX data not available
        if np.isnan(adx[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below lower Donchian or ADX weakens (<20)
            if (low[i] <= lower[i] or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above upper Donchian or ADX weakens (<20)
            if (high[i] >= upper[i] or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and ADX trend filter
            if volume_filter and adx[i] > 25:
                # Long: break above upper Donchian during strong uptrend
                if high[i] > upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian during strong downtrend
                elif low[i] < lower[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals