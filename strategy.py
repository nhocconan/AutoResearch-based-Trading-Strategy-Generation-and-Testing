#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
    # Donchian breakouts capture momentum in both bull and bear markets.
    # ATR regime filter (ATR(14) > ATR(50)) ensures we only trade in volatile regimes.
    # Volume confirmation (current volume > 1.3 * 20-period MA) validates breakout strength.
    # Fixed position size of 0.25 to minimize fee churn and control drawdown.
    # Target: 75-150 total trades over 4 years (19-37/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[1:14])  # Seed with simple average
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR(50) using Wilder's smoothing
    atr_50 = np.zeros_like(tr)
    atr_50[49] = np.mean(tr[1:50])  # Seed with simple average
    for i in range(50, len(tr)):
        atr_50[i] = (atr_50[i-1] * 49 + tr[i]) / 50
    
    # ATR regime: volatile when ATR(14) > ATR(50) (increasing volatility)
    atr_regime = atr_14 > atr_50
    
    # Align 1d ATR regime to 4h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3 * 20-period MA
        volume_filter = volume[i] > 1.3 * volume_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above prior period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below prior period's low
        
        # Regime filter: only trade in increasing volatility regimes
        regime_filter = atr_regime_aligned[i] > 0.5
        
        # Entry conditions: breakout with volume and regime confirmation
        long_entry = long_breakout and volume_filter and regime_filter
        short_entry = short_breakout and volume_filter and regime_filter
        
        # Exit conditions: opposite breakout
        long_exit = short_breakout
        short_exit = long_breakout
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_atr_regime_volume_v1"
timeframe = "4h"
leverage = 1.0