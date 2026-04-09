#!/usr/bin/env python3
# 4h_donchian_20_volume_chop_regime_v2
# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian high in trending market (CHOP < 38.2) with volume > 1.5x average.
# Short when price breaks below Donchian low in trending market (CHOP < 38.2) with volume > 1.5x average.
# Uses 1d HTF for choppiness calculation to avoid look-ahead and ensure proper alignment.
# Discrete position sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_volume_chop_regime_v2"
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
    
    # 1d HTF data for choppiness regime (to avoid look-ahead and ensure proper alignment)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM for 1d
    up_move = np.diff(high_1d, prepend=np.nan)
    down_move = -np.diff(low_1d, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    period = 14
    tr_smoothed = wilder_smooth(tr, period)
    plus_dm_smoothed = wilder_smooth(plus_dm, period)
    minus_dm_smoothed = wilder_smooth(minus_dm, period)
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR) / (ATR * period)) / log10(period)
    # Using ATR from smoothed TR
    atr = tr_smoothed  # This is already the smoothed TR (ATR)
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    chop = 100 * np.log10(sum_tr / (atr * period)) / np.log10(period)
    
    # Align choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR choppiness becomes too high (range market)
            if close[i] < lowest_low_20[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR choppiness becomes too high (range market)
            if close[i] > highest_high_20[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and trending market (CHOP < 38.2)
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            trending_market = chop_aligned[i] < 38.2
            
            if volume_confirmed and trending_market:
                # Long: price breaks above Donchian high
                if close[i] > highest_high_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close[i] < lowest_low_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals