#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout + 1w ADX trend filter + volume confirmation
# - Primary signal: 6h price breaks above/below Donchian(20) channel
# - Trend filter: 1w ADX(14) > 25 ensures we only trade in strong trends (works in bull/bear)
# - Volume confirmation: 6h volume > 1.5x 20-period average volume (avoid low-participation breakouts)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - ADX filter prevents whipsaws in ranging markets, Donchian captures breakouts with momentum
# - Volume confirmation ensures breakouts have institutional participation

name = "6h_1w_donchian_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w ADX(14) for trend strength
    # ADX calculation: +DM, -DM, TR, then smoothed
    high_diff = np.diff(high_1w, prepend=high_1w[0])
    low_diff = np.diff(low_1w, prepend=low_1w[0])
    
    plus_dm = np.where((high_diff > -low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((-low_diff > high_diff) & (-low_diff > 0), -low_diff, 0.0)
    
    tr1 = np.abs(np.diff(high_1w, prepend=high_1w[0]))
    tr2 = np.abs(np.diff(low_1w, prepend=low_1w[0]))
    tr3 = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(true_range, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    plus_di_14 = np.where(tr_14 != 0, (plus_dm_14 / tr_14) * 100, 0)
    minus_di_14 = np.where(tr_14 != 0, (minus_dm_14 / tr_14) * 100, 0)
    dx_14 = np.where((plus_di_14 + minus_di_14) != 0, 
                     np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 0)
    adx_14 = wilders_smoothing(dx_14, 14)
    
    # Align 1w ADX to 6h timeframe (completed 1w bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_14)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian(20) channel
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume confirmation: volume > 1.5x 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR ADX < 20 (trend weakening)
            if close[i] < lowest_low_20[i] or adx_aligned[i] < 20.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR ADX < 20 (trend weakening)
            if close[i] > highest_high_20[i] or adx_aligned[i] < 20.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and ADX filter
            # Long: price breaks above Donchian upper AND volume confirmation AND ADX > 25
            if close[i] > highest_high_20[i] and volume_confirmation[i] and adx_aligned[i] > 25.0:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower AND volume confirmation AND ADX > 25
            elif close[i] < lowest_low_20[i] and volume_confirmation[i] and adx_aligned[i] > 25.0:
                position = -1
                signals[i] = -0.25
    
    return signals