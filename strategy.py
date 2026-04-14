#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily ADX for trend strength and daily Bollinger Bands for mean reversion.
# Long when ADX > 25 (trending) and price touches lower Bollinger Band (oversold in uptrend).
# Short when ADX > 25 (trending) and price touches upper Bollinger Band (overbought in downtrend).
# Uses Bollinger Band width to filter ranging markets (avoid when BB width < 5th percentile).
# Volume confirmation (>1.3x 20-period average) reduces false signals.
# Designed to capture trend continuations after pullbacks in both bull and bear markets.
# Target: 25-40 trades/year per symbol (100-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for ADX and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_period = wilders_smoothing(tr, period)
    dm_plus_period = wilders_smoothing(dm_plus, period)
    dm_minus_period = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_period != 0, 100 * dm_plus_period / tr_period, 0)
    di_minus = np.where(tr_period != 0, 100 * dm_minus_period / tr_period, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    
    # Bollinger Band width for ranging filter
    bb_width = (upper - lower) / sma
    # Use 5th percentile as threshold for ranging markets
    bb_width_threshold = np.nanpercentile(bb_width[~np.isnan(bb_width)], 5) if np.sum(~np.isnan(bb_width)) > 0 else 0.01
    
    # Align indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need ADX and Bollinger Bands
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Avoid ranging markets (low BB width)
        not_ranging = bb_width_aligned[i] > bb_width_threshold
        
        if position == 0:
            # Look for entries in trending markets
            if adx_aligned[i] > 25 and volume_confirmed and not_ranging:
                # Long: price touches lower Bollinger Band (oversold in uptrend)
                if close[i] <= lower_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: price touches upper Bollinger Band (overbought in downtrend)
                elif close[i] >= upper_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle of Bollinger Bands or ADX weakens
            middle = (upper_aligned[i] + lower_aligned[i]) / 2
            if (close[i] >= middle or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle of Bollinger Bands or ADX weakens
            middle = (upper_aligned[i] + lower_aligned[i]) / 2
            if (close[i] <= middle or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_ADX_Bollinger_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0