#!/usr/bin/env python3
# Hypothesis: 12h Williams %R reversal with 1d ADX trend filter and volume confirmation.
# Enters long when Williams %R crosses above -80 from oversold with 1d bullish trend (ADX>25 and +DI>-DI) and volume > 1.5x MA20.
# Enters short when Williams %R crosses below -20 from overbought with 1d bearish trend (ADX>25 and +DI<+DI) and volume > 1.5x MA20.
# Exits when Williams %R returns to -50 (mean reversion) or opposing signal occurs.
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~12-37/year) to work in both bull and bear markets by requiring strong volume confirmation and trend alignment.
# Williams %R is effective in ranging markets (common in 2025 BTC/ETH bear/range) while ADX filter avoids whipsaws in strong trends.

name = "12h_WilliamsR_Reversal_1dADX_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1d data for ADX and DI (14-period)
    # True Range
    tr1 = pd.Series(high_1d - low_1d).values
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1))).values
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1))).values
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    # Smooth TR, DM+ and DM- with Wilder's smoothing (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_minus)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or \
           np.isnan(di_plus_aligned[i]) or np.isnan(di_minus_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from oversold with 1d bullish trend and volume spike
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                adx_aligned[i] > 25 and di_plus_aligned[i] > di_minus_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from overbought with 1d bearish trend and volume spike
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  adx_aligned[i] > 25 and di_plus_aligned[i] < di_minus_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to -50 (mean reversion) or bearish crossover
            if williams_r_aligned[i] >= -50 or (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to -50 (mean reversion) or bullish crossover
            if williams_r_aligned[i] <= -50 or (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals