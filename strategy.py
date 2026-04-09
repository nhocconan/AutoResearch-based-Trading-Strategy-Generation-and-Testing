#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter + volume confirmation
# - Primary signal: Elder Ray (Bull/Bear Power) from 13-period EMA
#   Bull Power = High - EMA13, Bear Power = Low - EMA13
#   Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling
# - Trend filter: 1d ADX > 25 to ensure we only trade in trending markets
# - Volume confirmation: 6h volume > 20-period median volume
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: ADX filter avoids ranging markets, Elder Ray captures momentum in trends

name = "6h_1d_elderray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    atr = wilders_smoothing(tr, period_adx)
    dm_plus_smooth = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smooth = wilders_smoothing(dm_minus, period_adx)
    
    # Avoid division by zero
    dm_plus_smooth = np.where(atr == 0, 0, dm_plus_smooth)
    dm_minus_smooth = np.where(atr == 0, 0, dm_minus_smooth)
    
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    dx = np.where((di_plus + di_minus) == 0, 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus))
    adx = wilders_smoothing(dx, period_adx)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute Elder Ray on 6h timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high_6h - ema_13  # Bull Power = High - EMA13
    bear_power = low_6h - ema_13   # Bear Power = Low - EMA13
    
    # Align Elder Ray components to primary timeframe (completed 6h bar only)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    ema_13_aligned = align_htf_to_ltf(prices, df_6h, ema_13)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_13_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bull Power turns negative OR ADX < 20 (trend weakening) OR volume drops
            if bull_power_aligned[i] <= 0 or adx_aligned[i] < 20 or not volume_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR ADX < 20 (trend weakening) OR volume drops
            if bear_power_aligned[i] >= 0 or adx_aligned[i] < 20 or not volume_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray alignment with ADX filter and volume confirmation
            # Long: Bull Power > 0 AND rising (momentum building) AND ADX > 25 AND volume regime
            # Short: Bear Power < 0 AND falling (momentum building) AND ADX > 25 AND volume regime
            if (bull_power_aligned[i] > 0 and 
                bull_power_aligned[i] > bull_power_aligned[i-1] and  # Rising bull power
                adx_aligned[i] > 25 and 
                volume_regime[i]):
                position = 1
                signals[i] = 0.25
            elif (bear_power_aligned[i] < 0 and 
                  bear_power_aligned[i] < bear_power_aligned[i-1] and  # Falling bear power (more negative)
                  adx_aligned[i] > 25 and 
                  volume_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals