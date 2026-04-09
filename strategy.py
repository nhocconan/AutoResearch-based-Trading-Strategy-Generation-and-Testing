#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX trend filter + volume confirmation
# - Uses 6h Williams %R(14) for oversold/overbought extremes (long < -80, short > -20)
# - Uses 1d ADX(14) to filter only trending markets (ADX > 25) for breakout continuation
# - Uses 6h volume spike (> 1.5x 20-period average) to confirm momentum
# - Only takes long when Williams %R crosses above -80 in uptrend (1d ADX > 25)
# - Only takes short when Williams %R crosses below -20 in downtrend (1d ADX > 25)
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to avoid fee drag
# - Williams %R is effective in both bull and bear markets when combined with trend filter
# - Volume confirmation reduces false signals and increases edge

name = "6h_1d_williamsr_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ , DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed / period) + current_value
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr_1d > 0, (dm_plus_smooth / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d > 0, (dm_minus_smooth / atr_1d) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, -50)
    
    # 6h volume confirmation (> 1.5x 20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        is_trending = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit conditions: Williams %R crosses above -20 (overbought) or trend ends
            if williams_r[i] >= -20 or not is_trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R crosses below -80 (oversold) or trend ends
            if williams_r[i] <= -80 or not is_trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries: Williams %R crosses extreme levels with volume spike in trending market
            if (is_trending and volume_spike[i]):
                # Long: Williams %R crosses above -80 from below (oversold bounce in uptrend)
                if williams_r[i] > -80 and i > 0 and williams_r[i-1] <= -80:
                    position = 1
                    signals[i] = 0.25
                # Short: Williams %R crosses below -20 from above (overbought rejection in downtrend)
                elif williams_r[i] < -20 and i > 0 and williams_r[i-1] >= -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals