#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d ATR regime filter + volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) to filter ranging markets (ATR < 20-period SMA of ATR = low vol chop).
- Entry: Long when price breaks above Donchian upper(20) AND ATR regime filter AND volume > 1.5 * 12h volume MA(20);
         Short when price breaks below Donchian lower(20) AND ATR regime filter AND volume > 1.5 * 12h volume MA(20).
- Exit: Long exits when price touches Donchian midpoint (mean of upper/lower); Short exits when price touches midpoint.
- Signal size: 0.25 discrete to minimize fee churn.
- Donchian captures breakouts; ATR filter avoids whipsaws in low volatility; volume confirms conviction.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with regime filter reducing false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR regime: trending when current ATR > MA of ATR (avoid low vol chop)
    atr_regime = atr_1d > atr_ma_1d
    
    # Align ATR regime to 12h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # Donchian(20) on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Get 12h data for volume MA(20)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr_regime_aligned[i]) or 
            np.isnan(vol_ma_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime and volume filters
        regime_filter = atr_regime_aligned[i] > 0.5  # True if trending regime
        vol_confirm = curr_volume > 1.5 * vol_ma_12h[i]
        
        if position == 0:
            # Check for entry signals
            if regime_filter and vol_confirm:
                # Long: price breaks above Donchian upper
                if curr_close > donchian_upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian lower
                elif curr_close < donchian_lower[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price touches Donchian midpoint
            if curr_close >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price touches Donchian midpoint
            if curr_close <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0