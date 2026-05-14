#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and 1d volume spike confirmation.
# Long when price breaks above Donchian upper AND 1d ATR(14)/ATR(50) > 0.8 (low volatility regime) AND 1d volume > 1.5 * 20-period average volume.
# Short when price breaks below Donchian lower AND 1d ATR(14)/ATR(50) > 0.8 AND 1d volume > 1.5 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 4h timeframe with strict entry conditions.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h.

name = "4h_Donchian20_Breakout_1dATRRegime_1dVolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # Ratio of short-term to long-term volatility
    low_vol_regime = atr_ratio > 0.8  # Low volatility regime (regime filter)
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    # Calculate 1d volume confirmation filter (HTF)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Donchian channel (20-period) for primary timeframe
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(n):
        if i < 19:  # Need 20 periods for Donchian(20)
            donchian_upper[i] = np.nan
            donchian_lower[i] = np.nan
            donchian_mid[i] = np.nan
        else:
            donchian_upper[i] = np.max(high[i-19:i+1])
            donchian_lower[i] = np.min(low[i-19:i+1])
            donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(low_vol_regime_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper AND low volatility regime AND volume confirmation
            if (open_[i] <= donchian_upper[i] and close[i] > donchian_upper[i] and 
                low_vol_regime_aligned[i] > 0.5 and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower AND low volatility regime AND volume confirmation
            elif (open_[i] >= donchian_lower[i] and close[i] < donchian_lower[i] and 
                  low_vol_regime_aligned[i] > 0.5 and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals