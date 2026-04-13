#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 12h volume confirmation and chop regime filter
    # Long: price breaks above Donchian(20) upper + volume > 1.5x 20-period 12h avg + chop < 61.8 (trending)
    # Short: price breaks below Donchian(20) lower + volume > 1.5x 20-period 12h avg + chop < 61.8 (trending)
    # Exit: price returns to Donchian midpoint (mean reversion within channel)
    # Uses 4h primary timeframe for balance of frequency and cost
    # Target: 100-200 total trades over 4 years (25-50/year) to minimize fee drag
    # Works in bull/bear: chop filter avoids false breakouts in ranging markets
    
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
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.zeros(len(df_4h))
    
    # Get 12h data for volume and chop (MTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels on 4h data (20-period)
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    # Middle = (Upper + Lower) / 2
    high_roll_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_upper = high_roll_max
    donch_lower = low_roll_min
    donch_middle = (donch_upper + donch_lower) / 2
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 12h data (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(n)
    # Simplified: we use true range and rolling sum
    tr1 = np.maximum(
        np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - np.roll(close_12h, 1)[1:])),
        np.abs(low_12h[1:] - np.roll(close_12h, 1)[1:])
    )
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    atr1 = tr1
    atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    # n * ATR(14) where ATR(14) is average true range over 14 periods
    atr14_avg = pd.Series(atr1).rolling(window=14, min_periods=14).mean().values
    chop_denominator = 14 * atr14_avg
    chop_raw = np.where(
        (atr14 > 0) & (chop_denominator > 0),
        np.log10(atr14 / chop_denominator) / np.log10(14) * 100,
        50  # default to neutral when invalid
    )
    chop = chop_raw
    
    # Align all indicators to 4h timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    donch_middle_aligned = align_htf_to_ltf(prices, df_4h, donch_middle)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
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
        
        # Volume confirmation: current 12h volume > 1.5x 20-period 12h average
        curr_vol_12h = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        volume_confirmed = curr_vol_12h > 1.5 * vol_avg_20_aligned[i]
        
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

name = "4h_12h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0