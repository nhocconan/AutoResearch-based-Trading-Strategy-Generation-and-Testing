#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 1d ADX Regime Filter + Volume Spike
# Williams %R identifies overbought/oversold conditions. In ranging markets (ADX < 25),
# we fade extremes: long when %R < -80, short when %R > -20. In trending markets (ADX >= 25),
# we only take pullbacks: long on %R < -50 in uptrend, short on %R > -50 in downtrend.
# Volume confirmation (>1.5x 20-period EMA volume) ensures participation.
# Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "6h_WilliamsR_MeanReversion_1dADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R (14-period) on 6h timeframe
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0, 
                      -100 * (highest_high - close) / (highest_high - lowest_low), -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ADX < 25 = ranging, ADX >= 25 = trending
            if adx_aligned[i] < 25:  # Ranging market: fade extremes
                # Long: oversold (%R < -80) + volume
                if wr[i] < -80 and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short: overbought (%R > -20) + volume
                elif wr[i] > -20 and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:  # Trending market: pullbacks only
                # Determine trend direction from ADX components (simplified: use price vs EMA)
                ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
                if close[i] > ema_20[i]:  # Uptrend
                    # Long: pullback to %R < -50 + volume
                    if wr[i] < -50 and volume_confirm:
                        signals[i] = 0.25
                        position = 1
                else:  # Downtrend
                    # Short: pullback to %R > -50 + volume
                    if wr[i] > -50 and volume_confirm:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: %R > -20 (overbought) OR ADX regime shift to ranging + %R > -50
            if wr[i] > -20 or (adx_aligned[i] < 25 and wr[i] > -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: %R < -80 (oversold) OR ADX regime shift to ranging + %R < -50
            if wr[i] < -80 or (adx_aligned[i] < 25 and wr[i] < -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals