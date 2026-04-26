#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRVolume_Regime_v2
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based regime filter.
- Long when price breaks above 20-period high with volume spike and low ATR ratio (low volatility regime)
- Short when price breaks below 20-period low with volume spike and low ATR ratio
- Volume spike confirms institutional participation
- ATR ratio filter avoids high volatility choppy markets where breakouts fail
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 75-200 trades over 4 years (19-50/year) for BTC/ETH/SOL
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility regime filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period ATR average (regime filter)
    atr_ma50 = pd.Series(atr14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr14 / np.where(atr_ma50 == 0, 1e-10, atr_ma50)
    
    # Volume spike (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for ATR MA, 20 for Donchian)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(atr_ratio[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions with volume confirmation and ATR regime filter
        price_above_high = close[i] > highest_high_20[i]
        price_below_low = close[i] < lowest_low_20[i]
        
        # Low volatility regime: ATR ratio < 1.2 (avoid high volatility chop)
        low_volatility_regime = atr_ratio[i] < 1.2
        
        if position == 0:
            # Long: break above upper band AND volume spike AND low volatility regime
            if price_above_high and volume_spike[i] and low_volatility_regime:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band AND volume spike AND low volatility regime
            elif price_below_low and volume_spike[i] and low_volatility_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below midpoint OR volatility increases
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < midpoint or not low_volatility_regime:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above midpoint OR volatility increases
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > midpoint or not low_volatility_regime:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_ATRVolume_Regime_v2"
timeframe = "4h"
leverage = 1.0