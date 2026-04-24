#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d ADX regime filter and volume confirmation.
- Long when Williams %R crosses above -80 (oversold reversal) AND 1d ADX < 25 (low volatility regime) AND volume spike
- Short when Williams %R crosses below -20 (overbought reversal) AND 1d ADX < 25 (low volatility regime) AND volume spike
- Exit when Williams %R crosses above -20 (for longs) or below -80 (for shorts) or regime changes
- Uses 6h primary with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Williams %R captures mean reversion in ranging markets; ADX filter avoids trending markets where reversals fail
- Designed for BTC/ETH ranging markets (2022-2024, 2025+) with mean reversion edges
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R on 6h data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX components on daily data
    df_1d_copy = df_1d.copy()
    df_1d_copy['high'] = df_1d['high'].values
    df_1d_copy['low'] = df_1d['low'].values
    df_1d_copy['close'] = df_1d['close'].values
    
    # True Range
    tr1 = df_1d_copy['high'] - df_1d_copy['low']
    tr2 = np.abs(df_1d_copy['high'] - df_1d_copy['close'].shift(1))
    tr3 = np.abs(df_1d_copy['low'] - df_1d_copy['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    up_move = df_1d_copy['high'] - df_1d_copy['high'].shift(1)
    down_move = df_1d_copy['low'].shift(1) - df_1d_copy['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])  # First value is simple average
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    # Calculate smoothed TR, +DM, -DM
    atr = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di = np.where(atr != 0, (plus_dm_smooth / atr) * 100, 0)
    minus_di = np.where(atr != 0, (minus_dm_smooth / atr) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 
                  0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Regime filter: low volatility (ADX < 25) for mean reversion
    low_vol_regime = adx_aligned < 25
    
    # Volume confirmation: volume > 1.8 * 20-period average (moderate spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 30, 20)  # Williams %R (14), ADX components, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) AND low vol regime AND volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and low_vol_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) AND low vol regime AND volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and low_vol_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) or regime changes to high vol
            if williams_r[i] >= -20 or not low_vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) or regime changes to high vol
            if williams_r[i] <= -80 or not low_vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0