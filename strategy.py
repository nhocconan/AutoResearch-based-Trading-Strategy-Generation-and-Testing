#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
    # Long: price breaks above Donchian upper + volume > 1.5x 20-period 1d avg + chop < 61.8 (trending)
    # Short: price breaks below Donchian lower + volume > 1.5x 20-period 1d avg + chop < 61.8 (trending)
    # Exit: price returns to Donchian midpoint (mean reversion in choppy markets)
    # Uses 4h timeframe for primary signals, 1d for volume/chop regime context
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
    # Works in bull/bear: chop filter avoids false breakouts in ranging markets, volume confirms institutional participation
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    vol_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.zeros_like(close_4h)
    
    # Get 1d data for volume and chop (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels on 4h data (20-period)
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    # Middle = (Upper + Lower) / 2
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    donch_middle = (donch_upper + donch_lower) / 2
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 1d data (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(n)
    # True range calculation
    tr1 = np.maximum(
        np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])),
        np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    )
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    atr1 = tr1
    atr14_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    atr14_avg = pd.Series(atr1).rolling(window=14, min_periods=14).mean().values
    chop_denominator = 14 * atr14_avg
    chop_raw = np.where(
        (atr14_sum > 0) & (chop_denominator > 0),
        np.log10(atr14_sum / chop_denominator) / np.log10(14) * 100,
        50  # default to neutral when invalid
    )
    chop = chop_raw
    
    # Align all indicators to 4h timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    donch_middle_aligned = align_htf_to_ltf(prices, df_4h, donch_middle)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(donch_middle_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period 1d average
        curr_vol_4h = vol_4h[i] if i < len(vol_4h) else 0
        volume_confirmed = curr_vol_4h > 1.5 * vol_avg_20_aligned[i]
        
        # Chop filter: trending market (chop < 61.8)
        is_trending = chop_aligned[i] < 61.8
        
        # Breakout conditions
        breakout_long = (close[i] > donch_upper_aligned[i] and 
                        volume_confirmed and 
                        is_trending)
        breakout_short = (close[i] < donch_lower_aligned[i] and 
                         volume_confirmed and 
                         is_trending)
        
        # Exit conditions: return to Donchian midpoint
        exit_long = position == 1 and close[i] <= donch_middle_aligned[i]
        exit_short = position == -1 and close[i] >= donch_middle_aligned[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0