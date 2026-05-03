#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX regime filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND 1d ADX < 25 (range market) AND volume > 1.5x 20-period MA.
# Short when Williams %R > -20 (overbought) AND 1d ADX < 25 (range market) AND volume > 1.5x 20-period MA.
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR ADX > 25 (trend regime).
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.
# This strategy focuses on BTC and ETH as primary targets, exploiting mean reversion in range markets
# while avoiding trend regimes where Williams %R fails. The 1d ADX filter ensures we only trade
# when higher timeframe is ranging, increasing win rate in choppy conditions like 2025 BTC/ETH.

name = "6h_WilliamsR_1dADX25_Volume"
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
    
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    dm_plus14 = wilders_smoothing(dm_plus, 14)
    dm_minus14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr14 > 0, (dm_plus14 / tr14) * 100, 0)
    di_minus = np.where(tr14 > 0, (dm_minus14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R on 6h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, -50)
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        williams_r_val = williams_r[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade in ranging markets (ADX < 25)
        in_range = adx_val < 25
        
        # Entry logic
        if position == 0:
            # Long: oversold AND ranging market AND volume spike
            if williams_r_val < -80 and in_range and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: overbought AND ranging market AND volume spike
            elif williams_r_val > -20 and in_range and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 OR ADX > 25 (trend regime)
            if williams_r_val > -50 or adx_val >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 OR ADX > 25 (trend regime)
            if williams_r_val < -50 or adx_val >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals