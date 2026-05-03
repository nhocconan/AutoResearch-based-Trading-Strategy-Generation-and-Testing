#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR regime filter + volume confirmation
# Long when price breaks above Donchian(20) high + ATR(14) > ATR(50) (expanding volatility) + volume spike
# Short when price breaks below Donchian(20) low + ATR(14) > ATR(50) + volume spike
# Uses 1d ATR regime to avoid whipsaw in ranging markets and capture true breakouts
# Designed for low trade frequency (19-50/year) to minimize fee drag. Works in both bull and bear markets.

name = "4h_Donchian20_ATRRegime_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) and ATR(50) on 1d
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_50 = tr.rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: expanding volatility when ATR(14) > ATR(50)
    atr_regime = atr_14 > atr_50
    
    # Align 1d ATR regime to 4h timeframe (wait for completed 1d bar)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Donchian(20) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (2.0x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(20, 50, 20) + 1  # Donchian(20) + ATR(50) + volume MA(20) + 1 for shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_regime_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + ATR regime + volume spike
            if (close[i] > donchian_high[i] and atr_regime_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + ATR regime + volume spike
            elif (close[i] < donchian_low[i] and atr_regime_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR ATR regime changes to contracting
            if (close[i] < donchian_low[i] or not atr_regime_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR ATR regime changes to contracting
            if (close[i] > donchian_high[i] or not atr_regime_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals