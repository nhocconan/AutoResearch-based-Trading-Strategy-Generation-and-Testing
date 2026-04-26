#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRVolume_Regime_v1
Hypothesis: 4h Donchian channel breakout with ATR-based volatility filter and volume confirmation.
- Long when price breaks above 20-period Donchian high with volume spike and ATR expansion
- Short when price breaks below 20-period Donchian low with volume spike and ATR expansion
- Uses ATR regime filter: only trade when ATR(14) > ATR(50) (expanding volatility)
- Designed for 15-35 trades/year (60-140 total over 4 years) to minimize fee drag
- Works in bull/bear markets by capturing breakouts during volatile periods
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for ATR(50)
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(14) and ATR(50) for volatility regime
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # First bar TR
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    tr50 = np.maximum(high - low, np.absolute(high - np.roll(close, 50)))
    tr50 = np.maximum(tr50, np.absolute(low - np.roll(close, 50)))
    tr50[:50] = high[:50] - low[:50]  # First 50 bars TR
    atr50 = pd.Series(tr50).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: expanding volatility (ATR14 > ATR50)
    atr_regime = atr14 > atr50
    
    # Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for ATR50 and EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr14[i]) or np.isnan(atr50[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(ema50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donch_low[i-1]  # Break below previous period's low
        
        # 1d trend filter
        trend_up = close[i] > ema50_1d_aligned[i]
        trend_down = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: breakout up + volume spike + ATR expansion + 1d uptrend
            if breakout_up and volume_spike[i] and atr_regime[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + volume spike + ATR expansion + 1d downtrend
            elif breakout_down and volume_spike[i] and atr_regime[i] and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below Donchian low OR 1d trend turns down
            if close[i] < donch_low[i] or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high OR 1d trend turns up
            if close[i] > donch_high[i] or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_ATRVolume_Regime_v1"
timeframe = "4h"
leverage = 1.0