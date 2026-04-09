#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v2
# Hypothesis: 4h Donchian channel breakout with volume confirmation and chop regime filter.
# Long when price breaks above Donchian(20) high with volume > 1.5x average and chop < 61.8 (trending).
# Short when price breaks below Donchian(20) low with volume > 1.5x average and chop < 61.8 (trending).
# Exit when price closes back inside Donchian(10) channel or opposite breakout occurs.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture strong trending moves while avoiding range-bound false breakouts.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Chop regime filter (14-period) - calculates on LTF but uses HTF for regime?
    # Actually, we'll calculate chop on 4h directly as it's our primary timeframe
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Chop calculation: sum of TR over period / (max(high) - min(low)) over same period
    sum_tr = tr.rolling(window=14, min_periods=14).sum().values
    max_high = high_s.rolling(window=14, min_periods=14).max().values
    min_low = low_s.rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (max_high - min_low + 1e-10)) / np.log10(14)
    
    # Donchian channels
    donchian_high_20 = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_s.rolling(window=20, min_periods=20).min().values
    donchian_high_10 = high_s.rolling(window=10, min_periods=10).max().values
    donchian_low_10 = low_s.rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(open_[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop filter: trending market (chop < 61.8)
        trending_regime = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: Price closes back below Donchian(10) low or opposite breakout
            if close[i] < donchian_low_10[i] or close[i] < donchian_low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes back above Donchian(10) high or opposite breakout
            if close[i] > donchian_high_10[i] or close[i] > donchian_high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume and regime confirmation
            bullish_breakout = (close[i] > donchian_high_20[i]) and volume_confirmed and trending_regime
            bearish_breakout = (close[i] < donchian_low_20[i]) and volume_confirmed and trending_regime
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals