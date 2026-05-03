#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Long when price breaks above Donchian upper (20-period high), ATR(1d) > 20-period ATR mean (high volatility regime), volume > 1.5x 20-bar average
# Short when price breaks below Donchian lower (20-period low), ATR(1d) > 20-period ATR mean, volume > 1.5x 20-bar average
# Exit when price returns to Donchian midpoint (mean reversion within the channel)
# Designed for low trade frequency (~20-50/year on 4h) to minimize fee drag
# Works in bull (breakouts with rising volume in high vol) and bear (breakdowns with rising volume in high vol) markets
# Uses Donchian channels for structure, 1d ATR for regime filter (avoids low vol whipsaws), volume for momentum confirmation

name = "4h_Donchian20_Volume_ATRRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    atr_regime = atr_14_1d > atr_ma_1d  # high volatility regime
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Calculate Donchian channels (20-period) on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume confirmation (1.5x 20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(20, 14) + 1  # Donchian(20) + ATR(14) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(atr_regime_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper, high volatility regime, volume spike
            if (close[i] > donch_high[i] and 
                atr_regime_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian lower, high volatility regime, volume spike
            elif (close[i] < donch_low[i] and 
                  atr_regime_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian midpoint (mean reversion)
            if close[i] < donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian midpoint (mean reversion)
            if close[i] > donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals