#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation
# - Primary signal: 6h price breaks above Camarilla R4 or below S4 from prior 1d session
# - Trend filter: 1d ADX > 25 ensures we only trade in trending markets (avoids chop)
# - Volume confirmation: 6h volume > 1.5x 20-period EMA of volume (avoids low-participation breakouts)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Camarilla breakouts capture momentum in trends, ADX filter ensures we avoid false breakouts in ranges

name = "6h_1d_camarilla_breakout_adx_volume_v1"
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
    
    # Pre-compute 1d ADX for trend filter (ADX > 25 = trending)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(tr_smooth != 0, tr_smooth, 1e-10)
    di_minus = 100 * dm_minus_smooth / np.where(tr_smooth != 0, tr_smooth, 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), 1e-10)
    adx = wilders_smoothing(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute prior day's Camarilla levels (using prior 1d bar to avoid look-ahead)
    # Camarilla levels based on prior day's OHLC
    close_prev = np.roll(close_1d, 1)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev[0] = close_1d[0]  # First bar: use current
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    r4 = pivot + (range_prev * 1.1 / 2)
    s4 = pivot - (range_prev * 1.1 / 2)
    r3 = pivot + (range_prev * 1.1 / 4)
    s3 = pivot - (range_prev * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Pre-compute 6h volume regime: volume > 1.5x 20-period EMA of volume
    volume = prices['volume'].values
    volume_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_regime = volume > (1.5 * volume_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below R3 (profit taking) OR ADX drops below 20 (trend weakening)
            if prices['close'].iloc[i] < r3_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above S3 (profit taking) OR ADX drops below 20 (trend weakening)
            if prices['close'].iloc[i] > s3_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and ADX filter
            # Long: price breaks above R4 AND volume regime AND ADX > 25
            if (prices['close'].iloc[i] > r4_aligned[i] and 
                volume_regime[i] and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 AND volume regime AND ADX > 25
            elif (prices['close'].iloc[i] < s4_aligned[i] and 
                  volume_regime[i] and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals