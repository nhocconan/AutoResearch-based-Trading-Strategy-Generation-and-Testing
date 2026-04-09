#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume regime and ADX trend filter
# - Uses 4h Donchian breakout from prior 20-bar structure for entries
# - Volume regime: 1d volume > 20-period median volume to ensure institutional participation
# - ADX trend filter: ADX(14) > 25 on 1d timeframe to confirm trending market
# - In trending markets (ADX > 25), only trade breakouts in direction of 1d EMA50 trend
# - In ranging markets (ADX <= 25), trade breakouts in either direction (mean reversion at extremes)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts, ADX filter avoids whipsaws in ranging markets

name = "4h_1d_donchian_vol_adx_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d True Range for ADX
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr_1d[0]  # First value
    
    # Calculate 1d ADX components
    plus_dm_1d = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm_1d = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Handle first values
    plus_dm_1d[0] = 0
    minus_dm_1d[0] = 0
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm_1d, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm_1d, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Handle division by zero and invalid values
    adx_1d = np.where(np.isnan(adx_1d) | (plus_di_1d + minus_di_1d == 0), 0, adx_1d)
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_regime = volume_1d > median_volume_20
    
    # Align all 1d indicators to 4h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR 
            #         in trending market (ADX > 25), price closes below EMA50
            if adx_aligned[i] > 25:
                # In trending market: exit if trend reverses
                if close[i] < ema_50_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # In ranging market: exit at opposite Donchian level
                if close[i] <= lowest_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR
            #         in trending market (ADX > 25), price closes above EMA50
            if adx_aligned[i] > 25:
                # In trending market: exit if trend reverses
                if close[i] > ema_50_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # In ranging market: exit at opposite Donchian level
                if close[i] >= highest_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation
            # Long: price breaks above Donchian upper band AND volume regime
            if high[i] >= highest_high[i] and volume_regime_aligned[i]:
                # In trending market (ADX > 25): only long if price above EMA50 (trend continuation)
                # In ranging market (ADX <= 25): long if breakout occurs (mean reversion at extreme)
                if adx_aligned[i] > 25:
                    if close[i] > ema_50_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                else:
                    position = 1
                    signals[i] = 0.25
            # Short: price breaks below Donchian lower band AND volume regime
            elif low[i] <= lowest_low[i] and volume_regime_aligned[i]:
                # In trending market (ADX > 25): only short if price below EMA50 (trend continuation)
                # In ranging market (ADX <= 25): short if breakout occurs (mean reversion at extreme)
                if adx_aligned[i] > 25:
                    if close[i] < ema_50_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                else:
                    position = -1
                    signals[i] = -0.25
    
    return signals