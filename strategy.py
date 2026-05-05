#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Bollinger Band breakout with 12h volume spike and 4h ADX trend filter
# Long when price breaks above 1d Bollinger Upper Band (20,2) AND 12h volume > 2.0 * avg_volume(20) AND 4h ADX > 20
# Short when price breaks below 1d Bollinger Lower Band (20,2) AND 12h volume > 2.0 * avg_volume(20) AND 4h ADX > 20
# Exit when price crosses back through 1d Bollinger Middle Band (20-period SMA)
# Uses discrete sizing 0.25 to balance return and risk
# Bollinger Bands provide dynamic support/resistance that adapts to volatility
# Volume confirmation validates breakout strength
# ADX filter ensures we trade in trending regimes, reducing false breakouts in ranging markets
# Target: 80-160 total trades over 4 years (20-40/year) for 4h timeframe

name = "4h_1dBB_Breakout_12hVolSpike_4hADX20"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for BB
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands (20,2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    bb_middle = sma_20  # 20-period SMA
    
    # Align 1d Bollinger Bands to 4h timeframe (wait for completed 1d bar)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    
    # Get 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least one completed 12h bar
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h average volume (20-period)
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    avg_volume_20_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_20_12h)
    volume_spike = volume_12h_aligned > (2.0 * avg_volume_20_12h_aligned)
    
    # Get 4h data for ADX calculation (using primary timeframe data)
    # Calculate 4h ADX (14-period)
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average (skip first NaN in tr)
        result[period-1] = np.nansum(data[1:period])
        # Wilder's smoothing: previous * (period-1)/period + current/period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (period-1)/period + data[i]/period
        return result
    
    atr_4h = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_4h != 0, 100 * dm_plus_smooth / atr_4h, 0)
    di_minus = np.where(atr_4h != 0, 100 * dm_minus_smooth / atr_4h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_4h = wilders_smoothing(dx, 14)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or np.isnan(adx_4h[i]) or 
            np.isnan(volume_spike[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Bollinger Upper Band, volume spike, ADX > 20, in session
            if (close[i] > bb_upper_aligned[i] and 
                volume_spike[i] and 
                adx_4h[i] > 20.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Bollinger Lower Band, volume spike, ADX > 20, in session
            elif (close[i] < bb_lower_aligned[i] and 
                  volume_spike[i] and 
                  adx_4h[i] > 20.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Bollinger Middle Band
            if close[i] < bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Bollinger Middle Band
            if close[i] > bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals