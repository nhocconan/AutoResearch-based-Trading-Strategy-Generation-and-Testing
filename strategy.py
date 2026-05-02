#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and 1d volume spike
# Uses weekly Camarilla pivot levels (R3/S3) from 1w timeframe to establish directional bias
# Only takes longs when price > weekly R3 (bullish regime) and shorts when price < weekly S3 (bearish regime)
# Entry triggered by 6h Donchian breakout in direction of weekly trend with 1d volume confirmation (2x average)
# Includes chop regime filter from 1d timeframe (CHOP < 61.8 = trending) to avoid ranging markets
# Discrete position sizing (0.25) to minimize fee churn
# Target: 75-150 total trades over 4 years = 19-38/year for 6h timeframe
# Weekly pivot provides strong regime filter that works in both bull (buy dips above R3) and bear (sell rallies below S3)
# Volume spike ensures institutional participation in breakouts
# Designed for low trade frequency to overcome fee drag in 6h timeframe

name = "6h_Donchian20_1wCamarillaPivot_VolumeSpike_ChopFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for weekly Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on prior week OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Typical price for pivot calculation
    typical_1w = (high_1w + low_1w + close_1w) / 3
    
    # Weekly Camarilla levels (using prior week's range)
    range_1w = high_1w - low_1w
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    s3_1w = close_1w - (range_1w * 1.1 / 4)
    r4_1w = close_1w + (range_1w * 1.1 / 2)
    s4_1w = close_1w - (range_1w * 1.1 / 2)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar close)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Load 1d data ONCE before loop for volume spike and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike (2x 20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 1d Chopiness Index (14) - trending when < 38.2, ranging when > 61.8
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR14
    atr1 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop = 100 * log15(sum(ATR14)/ (max(high)-min(low)) over 14 periods)
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log15(atr1 * 14 / (max_high - min_low))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian channels (20-period) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when Chop < 61.8 (not strongly ranging)
        if chop_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price > weekly R3 (bullish regime) + Donchian breakout up + volume spike
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > highest_high[i] and 
                volume_spike_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price < weekly S3 (bearish regime) + Donchian breakout down + volume spike
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < lowest_low[i] and 
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower band or reverses below weekly S3
            if close[i] < lowest_low[i] or close[i] < s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper band or reverses above weekly R3
            if close[i] > highest_high[i] or close[i] > r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals