#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX(14) + Bollinger Bands(20,2) with mean reversion logic
# Long when price touches lower BB, ADX < 20 (low volatility regime), and volume > 1.2x average
# Short when price touches upper BB, ADX < 20, and volume > 1.2x average
# Exit when price crosses middle BB (20-period SMA) or ADX rises above 25 (trend begins)
# Uses ADX for regime detection (range vs trend) and Bollinger Bands for mean reversion entries
# Works in both bull and bear markets by focusing on range-bound conditions
# Targets 15-35 trades per year (60-140 over 4 years) for low fee drag

name = "6h_ADX_BB_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation (more stable than 6h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilder_smooth(dx, period)
    adx_1d = adx
    
    # Align ADX to 6t
    adx_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Bollinger Bands on 6h data
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    bb_middle = sma_20
    
    # Volume confirmation: current volume > 1.2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for ADX and BB
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_6h[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        adx_val = adx_6h[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        bb_middle_val = bb_middle[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price at or below lower BB, low volatility (ADX < 20), volume confirmation
            if close_val <= bb_lower_val and adx_val < 20 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price at or above upper BB, low volatility (ADX < 20), volume confirmation
            elif close_val >= bb_upper_val and adx_val < 20 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above middle BB or volatility increases (ADX > 25)
            if close_val >= bb_middle_val or adx_val > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below middle BB or volatility increases (ADX > 25)
            if close_val <= bb_middle_val or adx_val > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals