#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ADX trend filter + volume spike
# - Primary signal: 12h Donchian channel breakout (20-period high/low)
# - Trend filter: 1d ADX > 25 ensures we only trade in trending markets (works in bull/bear)
# - Volume confirmation: 12h volume > 1.5 * 20-period median volume (avoid low-participation breakouts)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Stoploss: exit when price re-enters the Donchian channel (close-based)
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - ADX filter reduces whipsaws in ranging markets, improving bear market performance

name = "12h_1d_donchian_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX for trend filter (ADX > 25 = trending market)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period]) if np.any(~np.isnan(data[1:period])) else np.nan
        # Rest is Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    tr_smooth = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 12h Donchian channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high: 20-period rolling max
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian low: 20-period rolling min
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to primary timeframe (completed 12h bar only)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume regime: volume > 1.5 * 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR ADX falls below 20 (trend weakening)
            if close[i] < donch_low_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR ADX falls below 20 (trend weakening)
            if close[i] > donch_high_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and ADX trend filter
            # Long: price breaks above Donchian high AND volume spike AND ADX > 25 (strong trend)
            if (close[i] > donch_high_aligned[i] and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume spike AND ADX > 25 (strong trend)
            elif (close[i] < donch_low_aligned[i] and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals