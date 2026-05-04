#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band Width regime + Donchian(20) breakout with volume confirmation
# Bollinger Band Width < 0.05 = low volatility squeeze (range regime) → mean reversion at bands
# Bollinger Band Width > 0.10 = high volatility expansion (trend regime) → breakout continuation
# In range regime: fade Donchian touches (short upper band, long lower band)
# In trend regime: breakout Donchian breaks (long upper break, short lower break)
# Volume confirmation filters false breakouts. Works in both bull/bear via regime adaptation.

name = "12h_BBW_Donchian_Regime_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on primary timeframe
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    bb_width = (upper_band - lower_band) / basis  # normalized bandwidth
    
    # Donchian channels (20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Regime filter: Bollinger Band Width
    # Low BW (< 0.05) = squeeze/range regime → mean reversion
    # High BW (> 0.10) = expansion/trend regime → breakout
    range_regime = bb_width < 0.05
    trend_regime = bb_width > 0.10
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(basis[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Range regime: mean reversion at bands
            if range_regime[i]:
                if close[i] >= upper_band[i] and volume[i] > vol_ema_20[i]:
                    signals[i] = -0.25  # short at upper band
                    position = -1
                elif close[i] <= lower_band[i] and volume[i] > vol_ema_20[i]:
                    signals[i] = 0.25   # long at lower band
                    position = 1
            # Trend regime: breakout continuation
            elif trend_regime[i]:
                if close[i] > donchian_high[i] and volume[i] > (1.5 * vol_ema_20[i]):
                    signals[i] = 0.25   # long breakout
                    position = 1
                elif close[i] < donchian_low[i] and volume[i] > (1.5 * vol_ema_20[i]):
                    signals[i] = -0.25  # short breakdown
                    position = -1
        elif position == 1:
            # Exit long: range regime + price at lower band OR trend regime + donchian low break
            if range_regime[i] and close[i] <= lower_band[i]:
                signals[i] = 0.0
                position = 0
            elif trend_regime[i] and close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: range regime + price at upper band OR trend regime + donchian high break
            if range_regime[i] and close[i] >= upper_band[i]:
                signals[i] = 0.0
                position = 0
            elif trend_regime[i] and close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals