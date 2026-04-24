#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 12h ADX regime filter and volume spike confirmation.
- Long when Williams %R < -80 (oversold) AND 12h ADX < 25 (low volatility regime) AND volume > 2.0 * 20-period average
- Short when Williams %R > -20 (overbought) AND 12h ADX < 25 (low volatility regime) AND volume > 2.0 * 20-period average
- Exit when Williams %R crosses above -50 (for long) or below -50 (for short)
- Uses 6h primary with 12h HTF to target 50-150 trades over 4 years (12-37/year)
- Williams %R identifies overextended moves; ADX filter ensures mean reversion works in ranging markets; volume spike confirms exhaustion
- Designed to work in ranging markets (common in 2025 BTC/ETH) with controlled trade frequency
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
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # Get 12h data ONCE before loop for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX(14)
    plus_dm = np.where((df_12h['high'].diff()) > (df_12h['low'].diff().abs()), 
                       np.maximum(df_12h['high'].diff(), 0), 0)
    minus_dm = np.where((df_12h['low'].diff().abs()) > (df_12h['high'].diff()), 
                        np.maximum(df_12h['low'].diff().abs(), 0), 0)
    tr = np.maximum(
        df_12h['high'] - df_12h['low'],
        np.maximum(
            abs(df_12h['high'] - df_12h['close'].shift(1)),
            abs(df_12h['low'] - df_12h['close'].shift(1))
        )
    )
    # Wilder's smoothing
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_12h = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_12h + 1e-10)
    minus_di_12h = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_12h + 1e-10)
    dx_12h = 100 * abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h + 1e-10)
    adx_12h = pd.Series(dx_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 12h ADX to 6h timeframe (waits for completed 12h bar)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Regime filter: low volatility (ADX < 25) for mean reversion
    low_vol_regime = adx_12h_aligned < 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need Williams %R, volume MA, and ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(adx_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: oversold AND low volatility regime AND volume spike
            if williams_r[i] < -80 and low_vol_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought AND low volatility regime AND volume spike
            elif williams_r[i] > -20 and low_vol_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (recovery from oversold)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (decline from overbought)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0