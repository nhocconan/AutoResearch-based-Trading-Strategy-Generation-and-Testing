#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d ADX Regime + Volume Spike
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Williams %R identifies overbought/oversold extremes: long when %R < -80, short when %R > -20
# 1d ADX regime filter: ADX > 25 for trending markets (trend follow), ADX < 20 for ranging markets (mean revert)
# Volume spike (2x 20-period average) confirms institutional participation
# Works in bull markets via trend continuation and bear markets via mean reversion in ranges
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d ADX (14-period) for regime filtering
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range components
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = np.abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = np.abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = np.where(
        (pd.Series(df_1d['high']) - pd.Series(df_1d['high']).shift(1)) > 
        (pd.Series(df_1d['low']).shift(1) - pd.Series(df_1d['low'])),
        np.maximum(pd.Series(df_1d['high']) - pd.Series(df_1d['high']).shift(1), 0),
        0
    ).values
    dm_minus = np.where(
        (pd.Series(df_1d['low']).shift(1) - pd.Series(df_1d['low'])) > 
        (pd.Series(df_1d['high']) - pd.Series(df_1d['high']).shift(1)),
        np.maximum(pd.Series(df_1d['low']).shift(1) - pd.Series(df_1d['low']), 0),
        0
    ).values
    
    # Smooth TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Regime-based entry logic
            if adx_aligned[i] > 25:  # Trending regime - trend following
                # Long: Williams %R oversold AND volume spike
                if williams_r[i] < -80 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R overbought AND volume spike
                elif williams_r[i] > -20 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif adx_aligned[i] < 20:  # Ranging regime - mean reversion
                # Long: Williams %R deeply oversold AND volume spike
                if williams_r[i] < -90 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R deeply overbought AND volume spike
                elif williams_r[i] > -10 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # Transition regime (20 <= ADX <= 25) - no trades
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to neutral territory OR ADX drops below 20 (regime change to range)
            if williams_r[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral territory OR ADX drops below 20 (regime change to range)
            if williams_r[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals