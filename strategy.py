#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d ATR regime filter + session filter (08-20 UTC)
# Uses 4h Donchian(20) for breakout signals: long when price > 20-period high, short when price < 20-period low
# 1d ATR(14) / ATR(50) ratio filter: only trade when ratio < 1.0 (low volatility regime) to avoid whipsaws
# Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity periods
# Works in bull/bear: Donchian breakouts capture trends, ATR filter avoids false breakouts in choppy markets
# Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing 0.20

name = "4h_1d_donchian_atr_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Rolling max/min for Donchian channels
    dh_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (completed 4h bars only)
    dh_20_aligned = align_htf_to_ltf(prices, df_4h, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_4h, dl_20)
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio: ATR(14)/ATR(50) < 1.0 indicates low volatility regime
    atr_ratio = np.full(n, np.nan)
    valid_atr = (~np.isnan(atr_14)) & (~np.isnan(atr_50)) & (atr_50 > 0)
    atr_ratio[valid_atr] = atr_14[valid_atr] / atr_50[valid_atr]
    
    # Align ATR ratio to 1h timeframe (completed 1d bars only)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(dh_20_aligned[i]) or np.isnan(dl_20_aligned[i]) or
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Low volatility regime filter: only trade when ATR ratio < 1.0
        low_vol_regime = atr_ratio_aligned[i] < 1.0
        
        if position == 1:  # Long position
            # Exit: price < 4h Donchian low (breakdown) OR high volatility regime
            if close[i] < dl_20_aligned[i] or not low_vol_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price > 4h Donchian high (breakout) OR high volatility regime
            if close[i] > dh_20_aligned[i] or not low_vol_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic: Donchian breakout + low volatility regime
            if low_vol_regime:
                # Long entry: price > 4h Donchian high (breakout)
                if close[i] > dh_20_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: price < 4h Donchian low (breakdown)
                elif close[i] < dl_20_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals