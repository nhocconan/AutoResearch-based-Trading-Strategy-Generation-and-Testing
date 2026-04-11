#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_regime_v1
# Strategy: 4h Donchian channel breakout with volume confirmation and 1d chop regime filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture trend continuations. Volume confirms breakout strength.
# Chop regime filter avoids whipsaws in ranging markets (CHOP > 61.8) and only takes breakouts
# in trending markets (CHOP < 38.2). Works in both bull (long breakouts) and bear (short breakdowns).
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

name = "4h_1d_donchian_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Chop regime (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of absolute price changes (|close - close_prev|)
    abs_changes = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_abs_changes = pd.Series(abs_changes).rolling(window=14, min_periods=14).sum().values
    
    # Chop = (sum of |close - close_prev| over n) / (sum of TR over n) * 100
    chop = np.where(atr > 0, (sum_abs_changes / (atr * 14)) * 100, 100)
    
    # Chop regime: trending if CHOP < 38.2, ranging if CHOP > 61.8
    chop_trending = chop < 38.2
    chop_ranging = chop > 61.8
    
    # Align chop regime to 4h
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging)
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_trending_aligned[i]) or np.isnan(chop_ranging_aligned[i]) or
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry logic: Donchian breakout + volume + chop regime (trending only)
        if vol_confirm[i] and chop_trending_aligned[i]:
            # Long breakout: price breaks above Donchian high
            if close[i] > donchian_high[i-1] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below Donchian low
            elif close[i] < donchian_low[i-1] and position != -1:
                position = -1
                signals[i] = -0.25
        
        # Exit: chop regime becomes ranging or opposite breakout
        elif position == 1 and (chop_ranging_aligned[i] or close[i] < donchian_low[i-1]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (chop_ranging_aligned[i] or close[i] > donchian_high[i-1]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Fix: align_htf_to_ltf not align_ltf_to_hlf
# Correcting the import and function call

#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_regime_v1
# Strategy: 4h Donchian channel breakout with volume confirmation and 1d chop regime filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture trend continuations. Volume confirms breakout strength.
# Chop regime filter avoids whipsaws in ranging markets (CHOP > 61.8) and only takes breakouts
# in trending markets (CHOP < 38.2). Works in both bull (long breakouts) and bear (short breakdowns).
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Chop regime (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of absolute price changes (|close - close_prev|)
    abs_changes = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_abs_changes = pd.Series(abs_changes).rolling(window=14, min_periods=14).sum().values
    
    # Chop = (sum of |close - close_prev| over n) / (sum of TR over n) * 100
    chop = np.where(atr > 0, (sum_abs_changes / (atr * 14)) * 100, 100)
    
    # Chop regime: trending if CHOP < 38.2, ranging if CHOP > 61.8
    chop_trending = chop < 38.2
    chop_ranging = chop > 61.8
    
    # Align chop regime to 4h
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging)
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_trending_aligned[i]) or np.isnan(chop_ranging_aligned[i]) or
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry logic: Donchian breakout + volume + chop regime (trending only)
        if vol_confirm[i] and chop_trending_aligned[i]:
            # Long breakout: price breaks above Donchian high
            if close[i] > donchian_high[i-1] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below Donchian low
            elif close[i] < donchian_low[i-1] and position != -1:
                position = -1
                signals[i] = -0.25
        
        # Exit: chop regime becomes ranging or opposite breakout
        elif position == 1 and (chop_ranging_aligned[i] or close[i] < donchian_low[i-1]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (chop_ranging_aligned[i] or close[i] > donchian_high[i-1]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals