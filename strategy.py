#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX trend filter + volume confirmation
# - Primary signal: Williams %R(14) on 6h for overbought/oversold conditions
# - Trend filter: 1d ADX(14) > 25 to ensure we trade in trending markets only
# - Volume confirmation: 6h volume > 1.3x 20-period average volume
# - Logic: In strong trends (ADX > 25), fade extreme Williams %R readings (mean reversion within trend)
# - Long: %R < -80 (oversold) + ADX > 25 + volume spike
# - Short: %R > -20 (overbought) + ADX > 25 + volume spike
# - Exit: %R returns to -50 level (mean reversion) or opposite extreme
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) per 6h strategy guidelines

name = "6h_1d_williamsr_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr_14 = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_14
    minus_di = 100 * minus_dm_smooth / atr_14
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 14:
        adx[13] = np.mean(dx[:14])  # First ADX is average of first 14 DX values
        for i in range(14, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14  # Wilder smoothing
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close_6h) / (highest_high - lowest_low),
                          -50)  # neutral when no range
    
    # Pre-compute 6h volume spike filter
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_6h > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns to -50 (mean reversion) or goes overbought
            if williams_r[i] >= -50 or williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to -50 (mean reversion) or goes oversold
            if williams_r[i] <= -50 or williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for extreme Williams %R readings with ADX trend filter and volume confirmation
            if adx_aligned[i] > 25 and volume_spike[i]:  # Strong trend + volume confirmation
                # Long: oversold condition (%R < -80)
                if williams_r[i] < -80:
                    position = 1
                    signals[i] = 0.25
                # Short: overbought condition (%R > -20)
                elif williams_r[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals