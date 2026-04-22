# 3. If a symbol fails train, its test is skipped. Symbols are independent.

#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian channel breakout with 1-day ATR filter and volume confirmation.
Trades breakouts above/below the 20-period Donchian channel only when:
1. The breakout aligns with the daily ATR-based volatility regime (high ATR = trending)
2. Volume confirms institutional participation (volume > 1.5x 20-period average)
Designed for low trade frequency (20-50 trades/year) to minimize fee drag and work in both
bull and bear markets by filtering breakouts with volatility regime and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channel (20-period) - primary timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Load daily data for ATR filter and volatility regime - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily ATR (14-period) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf  # First value has no previous close
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_ma = pd.Series(atr_14).rolling(window=10, min_periods=10).mean().values  # ATR trend
    
    # Align daily ATR and ATR trend to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_14_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_14_ma)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_14_ma_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime: current ATR above its moving average (trending market)
        vol_regime = atr_14_aligned[i] > atr_14_ma_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and vol_regime and vol_confirmed:
            # Long: breakout above Donchian upper band
            if close[i] > donchian_upper[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian lower band
            elif close[i] < donchian_lower[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle of Donchian channel or volatility regime changes
            exit_signal = False
            
            if position == 1:
                # Exit long: price closes below midpoint of Donchian channel
                midpoint = (donchian_upper[i] + donchian_lower[i]) / 2
                if close[i] < midpoint:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price closes above midpoint of Donchian channel
                midpoint = (donchian_upper[i] + donchian_lower[i]) / 2
                if close[i] > midpoint:
                    exit_signal = True
            
            # Also exit if volatility regime shifts to ranging (low ATR)
            if not vol_regime:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_ATR14_Volume_Breakout"
timeframe = "4h"
leverage = 1.0