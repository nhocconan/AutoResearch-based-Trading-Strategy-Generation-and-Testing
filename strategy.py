#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter.
# Long when price breaks above 20-period Donchian high AND 1d volume > 1.5x 20-period MA AND 1d chop < 61.8 (trending regime).
# Short when price breaks below 20-period Donchian low AND 1d volume > 1.5x 20-period MA AND 1d chop < 61.8.
# Exit when price crosses the Donchian midline (10-period average of high/low) OR chop > 61.8 (range regime).
# Uses 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian channels provide clear trend structure, volume confirms participation, chop filter avoids whipsaws in ranging markets.

name = "12h_Donchian20_1dVolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period MA for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # ATR (14-period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum_atr_14 / (hh_14 - ll_14)) / log10(14)
    range_14 = hh_14 - ll_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop_1d = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high_20 + lowest_low_20) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 12h volume > 1.5x 20-period volume MA (using 1d volume as proxy)
        # Since we don't have 12h volume MA, we use 1d volume spike as confirmation of participation
        volume_spike = vol_ma_20_1d_aligned[i] > 0 and volume[i] > (vol_ma_20_1d_aligned[i] * 1.5)
        
        # Chop regime: < 61.8 = trending, > 61.8 = ranging
        chop_trending = chop_1d_aligned[i] < 61.8
        chop_ranging = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND trending regime AND session
            if close[i] > highest_high_20[i] and volume_spike and chop_trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND trending regime AND session
            elif close[i] < lowest_low_20[i] and volume_spike and chop_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle OR chop becomes ranging
            if close[i] < donchian_middle[i] or chop_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle OR chop becomes ranging
            if close[i] > donchian_middle[i] or chop_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals