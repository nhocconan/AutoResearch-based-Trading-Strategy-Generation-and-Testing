#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h volume confirmation and 1d chop regime filter
# Use 4h/1d for signal direction (trend/regime), 1h only for entry timing precision
# Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing 0.20
# Works in bull/bear: chop regime filter adapts, Donchian breakout captures strong moves
# Session filter (08-20 UTC) reduces noise trades

name = "1h_4h_1d_donchian_volume_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h timeframe (wait for 4h bar close)
    highest_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # Load 1d data ONCE before loop for volume and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - smoothed TR using Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align 1d indicators to 1h timeframe (wait for 1d bar close)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(highest_high_4h_aligned[i]) or np.isnan(lowest_low_4h_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Regime filter: CHOP < 50 = trending (follow breakout), CHOP > 50 = range (mean revert)
        trending_regime = chop_1d_aligned[i] < 50
        ranging_regime = chop_1d_aligned[i] > 50
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian lower band OR regime shifts to ranging
            if close[i] < lowest_low_4h_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian upper band OR regime shifts to ranging
            if close[i] > highest_high_4h_aligned[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow breakout in trending regime
                if close[i] > highest_high_4h_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.20
                elif close[i] < lowest_low_4h_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.20
            elif ranging_regime:
                # Mean revert at 4h Donchian bands in ranging regime
                if close[i] < lowest_low_4h_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.20
                elif close[i] > highest_high_4h_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.20
    
    return signals