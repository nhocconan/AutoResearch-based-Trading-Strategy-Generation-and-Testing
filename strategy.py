#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R for mean reversion signals in ranging markets, 
# filtered by 1d ADX to avoid trending conditions. Williams %R < -80 indicates oversold (long), 
# > -20 indicates overbought (short). ADX < 20 ensures range-bound conditions to avoid false signals.
# Volume confirmation (>1.5x 20-period average) reduces false breakouts.
# Designed to work in both bull and bear markets by focusing on mean reversion in ranges.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation (14-period)
    williams_period = 14
    highest_high = pd.Series(high_1d).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # ADX calculation on 1d data
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero when both DI are zero
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Williams %R signals: < -80 oversold (long), > -20 overbought (short)
    williams_long_signal = williams_r < -80
    williams_short_signal = williams_r > -20
    
    # Range filter: ADX < 20 indicates ranging market
    ranging = adx < 20
    
    # Align indicators to 4h timeframe
    williams_long_aligned = align_htf_to_ltf(prices, df_1d, williams_long_signal)
    williams_short_aligned = align_htf_to_ltf(prices, df_1d, williams_short_signal)
    ranging_aligned = align_htf_to_ltf(prices, df_1d, ranging)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(williams_period, 20)  # Need Williams %R and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_long_aligned[i]) or 
            np.isnan(williams_short_aligned[i]) or
            np.isnan(ranging_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for mean reversion signals in ranging markets
            # Long: Williams %R oversold AND ranging market AND volume confirmation
            if (williams_long_aligned[i] and 
                ranging_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought AND ranging market AND volume confirmation
            elif (williams_short_aligned[i] and 
                  ranging_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral zone (> -50) or trend emerges
            if (williams_r[i] > -50 or  # Williams %R returned to neutral
                adx[i] >= 25):  # Trend emerging
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral zone (< -50) or trend emerges
            if (williams_r[i] < -50 or  # Williams %R returned to neutral
                adx[i] >= 25):  # Trend emerging
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dWilliamsR_ADX_Range_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0