#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR-based volume confirmation + chop regime filter
# Donchian breakout captures momentum in trending markets
# ATR-normalized volume avoids false signals during low volatility
# Choppiness index regime filter adapts strategy: CHOP < 38.2 = trend following, CHOP > 61.8 = mean reversion
# Works in bull/bear: regime filter prevents whipsaw in ranging markets, breakout captures strong moves
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_1d_donchian_atr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) using Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = (1 - 1/period) * result[i-1] + (1/period) * values[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR-normalized volume ratio: volume / ATR
    # Higher ratio = volume spike relative to current volatility
    vol_atr_ratio_1d = np.where(atr_1d != 0, volume_1d / atr_1d, 0)
    avg_vol_atr_ratio_1d = pd.Series(vol_atr_ratio_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # ATR(14) - smoothed TR (already calculated as atr_1d)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align 1d indicators to 4h timeframe (wait for 1d bar close)
    avg_vol_atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_atr_ratio_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(avg_vol_atr_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume/ATR ratio > 1.5x 1d average volume/ATR ratio
        # Calculate current 4h ATR(14) for volume normalization
        if i >= 14:
            # TR for 4h
            tr1_4h = np.abs(high[i] - low[i])
            tr2_4h = np.abs(high[i] - close[i-1]) if i > 0 else np.inf
            tr3_4h = np.abs(low[i] - close[i-1]) if i > 0 else np.inf
            tr_4h = max(tr1_4h, tr2_4h, tr3_4h)
            # Simplified ATR approximation for volume normalization (using recent TR)
            atr_4h_approx = tr_4h  # Using current TR as proxy for simplicity
        else:
            atr_4h_approx = 1.0  # Avoid division by zero
        
        vol_atr_ratio_4h = volume[i] / atr_4h_approx if atr_4h_approx != 0 else 0
        volume_confirmed = vol_atr_ratio_4h > 1.5 * avg_vol_atr_ratio_1d_aligned[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (mean revert)
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR regime shifts to ranging
            if close[i] < lowest_low[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR regime shifts to ranging
            if close[i] > highest_high[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow breakout in trending regime
                if close[i] > highest_high[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_low[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at Donchian bands in ranging regime
                if close[i] < lowest_low[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > highest_high[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals