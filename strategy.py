#!/usr/bin/env python3
"""
6h_VolumeSpike_RegimeAdaptive_Donchian
Hypothesis: On 6h timeframe, combine Donchian breakout with volatility regime filter (ATR ratio) and volume spike confirmation.
In high volatility regime (ATR(7)/ATR(30) > 1.5), trade breakouts; in low volatility regime, fade at bands.
Volume spike (>2x 20 EMA volume) confirms institutional participation.
Designed for 50-150 total trades over 4 years with discrete position sizing (0.0, ±0.25).
Adapts to both trending and ranging markets via volatility regime detection.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility regime
    tr1 = pd.Series(high).rolling(window=2).max().values - pd.Series(low).rolling(window=2).min().values
    tr2 = abs(pd.Series(high).rolling(window=2).max().values - pd.Series(close).shift(1).rolling(window=2).min().values)
    tr3 = abs(pd.Series(low).rolling(window=2).min().values - pd.Series(close).shift(1).rolling(window=2).max().values)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr7 / (atr30 + 1e-10)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Regime logic: high volatility = breakout, low volatility = mean reversion
        if atr_ratio[i] > 1.5:  # High volatility regime - trade breakouts
            # Long breakout: price breaks above upper Donchian + volume spike
            if close[i] > period20_high[i] and volume_spike[i]:
                if position != 1:
                    signals[i] = base_size
                    position = 1
                else:
                    signals[i] = base_size
            # Short breakout: price breaks below lower Donchian + volume spike
            elif close[i] < period20_low[i] and volume_spike[i]:
                if position != -1:
                    signals[i] = -base_size
                    position = -1
                else:
                    signals[i] = -base_size
            # Exit: price returns to middle of channel
            elif position == 1 and close[i] < (period20_high[i] + period20_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > (period20_high[i] + period20_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = base_size
                else:
                    signals[i] = -base_size
        else:  # Low volatility regime - fade at bands
            # Long fade: price touches lower band + volume spike (contrarian)
            if close[i] <= period20_low[i] and volume_spike[i]:
                if position != 1:
                    signals[i] = base_size
                    position = 1
                else:
                    signals[i] = base_size
            # Short fade: price touches upper band + volume spike (contrarian)
            elif close[i] >= period20_high[i] and volume_spike[i]:
                if position != -1:
                    signals[i] = -base_size
                    position = -1
                else:
                    signals[i] = -base_size
            # Exit: price returns to middle of channel
            elif position == 1 and close[i] >= (period20_high[i] + period20_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] <= (period20_high[i] + period20_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = base_size
                else:
                    signals[i] = -base_size
    
    return signals

name = "6h_VolumeSpike_RegimeAdaptive_Donchian"
timeframe = "6h"
leverage = 1.0