#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Choppiness Index to detect trending vs ranging markets,
# combined with 4h Donchian breakout and volume confirmation.
# In trending markets (CHOP < 38.2): breakout entries in direction of trend.
# In ranging markets (CHOP > 61.8): mean reversion at Donchian channels.
# Volume > 1.5x 20-period average confirms breakout strength.
# Designed to work in both bull and bear markets by adapting to market regime.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR (14-period)
    atr_period = 14
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # High-Low range over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    # Choppiness Index: 100 * log10(sum(ATR) / (HH - LL)) / log10(14)
    # Avoid division by zero and log of zero
    ratio = np.where(range_hl > 0, sum_atr / range_hl, 1.0)
    chop = 100 * np.log10(ratio) / np.log10(14)
    chop = np.concatenate([[np.full(14, np.nan)], chop[14:]])  # First 14 values are NaN
    
    # Load 4h data ONCE for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian Channels on 4h data (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_period = 20
    upper_channel = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    upper_channel_aligned = align_htf_to_ltf(prices, df_4h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_4h, lower_channel)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20, 20)  # Need Chop (14+14+6), Donchian (20), volume MA (20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(upper_channel_aligned[i]) or
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Chop thresholds
        chopping = chop_aligned[i] > 61.8  # Ranging market
        trending = chop_aligned[i] < 38.2  # Trending market
        
        if position == 0:
            if trending and volume_confirmed:
                # In trending market: breakout in direction of trend
                # Determine trend direction using price vs midpoint
                mid_channel = (upper_channel_aligned[i] + lower_channel_aligned[i]) / 2
                if close[i] > mid_channel:
                    # Uptrend: long on breakout above upper channel
                    if close[i] > upper_channel_aligned[i]:
                        position = 1
                        signals[i] = position_size
                else:
                    # Downtrend: short on breakout below lower channel
                    if close[i] < lower_channel_aligned[i]:
                        position = -1
                        signals[i] = -position_size
            elif chopping and volume_confirmed:
                # In ranging market: mean reversion at channels
                if close[i] > upper_channel_aligned[i]:
                    # Sold off at resistance: short
                    position = -1
                    signals[i] = -position_size
                elif close[i] < lower_channel_aligned[i]:
                    # Bought at support: long
                    position = 1
                    signals[i] = position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle or opposite signal
            mid_channel = (upper_channel_aligned[i] + lower_channel_aligned[i]) / 2
            if close[i] < mid_channel:  # Return to middle
                position = 0
                signals[i] = 0.0
            elif chopping and close[i] < lower_channel_aligned[i]:  # Reverse signal in ranging
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle or opposite signal
            mid_channel = (upper_channel_aligned[i] + lower_channel_aligned[i]) / 2
            if close[i] > mid_channel:  # Return to middle
                position = 0
                signals[i] = 0.0
            elif chopping and close[i] > upper_channel_aligned[i]:  # Reverse signal in ranging
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dChop_4hDonchian_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0