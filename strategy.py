#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter
# - Williams %R(14) on 6h for overbought/oversold signals
# - 1d ADX(14) to filter ranging markets (ADX < 25) where mean reversion works best
# - Only take mean reversion trades when 1d ADX < 25 (ranging/low trend)
# - Long: Williams %R < -80 (oversold) and ADX < 25
# - Short: Williams %R > -20 (overbought) and ADX < 25
# - Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to avoid fee drag
# - Williams %R is effective in ranging markets which are common in bear/range regimes (2022-2024, 2025+)

name = "6h_1d_williamsr_adx_meanrev_v1"
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
    
    # 1d ADX(14) for regime filter
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
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: prev * (period-1)/period + current/period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = WilderSmooth(tr, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr_1d > 0, (dm_plus_smooth / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d > 0, (dm_minus_smooth / atr_1d) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = WilderSmooth(dx, 14)
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h Williams %R(14) for mean reversion signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, -50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(williams_r[i]) or
            highest_high[i] == lowest_low[i]):  # Avoid division by zero in Williams %R
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R crosses back above -50 (mean reversion complete)
            if williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R crosses back below -50 (mean reversion complete)
            if williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries in ranging markets (ADX < 25)
            if adx_1d_aligned[i] < 25:
                if williams_r[i] < -80:  # Oversold
                    position = 1
                    signals[i] = 0.25
                elif williams_r[i] > -20:  # Overbought
                    position = -1
                    signals[i] = -0.25
    
    return signals