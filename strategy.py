#!/usr/bin/env python3
# 4h_donchian_breakout_volume_regime_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter.
# Long: Price breaks above 4h Donchian(20) high, volume > 1.3x 20-period average, and choppy market (CHOP > 61.8).
# Short: Price breaks below 4h Donchian(20) low, volume > 1.3x 20-period average, and choppy market (CHOP > 61.8).
# Exit: Opposite Donchian breakout or ATR trailing stop (2.5x ATR from extreme).
# Uses 4h Donchian for structure, volume for confirmation, CHOP regime filter to avoid trending markets.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_regime_v2"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility and trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # 4h Donchian(20) channels
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (CHOP) regime filter
    # CHOP > 61.8 = choppy/range (good for mean reversion/breakout fade)
    # We use CHOP > 61.8 to filter for ranging markets where breakouts are more likely to fail and reverse
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = high_s.rolling(window=14, min_periods=14).max().values
    lowest_low = low_s.rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((highest_high - lowest_low) == 0, 100, chop)
    chop = np.where(np.isnan(chop), 100, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Regime filter: choppy market (CHOP > 61.8)
        regime_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if long_high > 0 and close[i] < long_high - 2.5 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price breaks below 4h Donchian low
            elif close[i] < donchian_low[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if short_low > 0 and close[i] > short_low + 2.5 * atr[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            # Exit: Price breaks above 4h Donchian high
            elif close[i] > donchian_high[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Donchian high breakout with volume and regime filter
            if (close[i] > donchian_high[i]) and volume_confirmed and regime_filter:
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            # Short entry: Donchian low breakout with volume and regime filter
            elif (close[i] < donchian_low[i]) and volume_confirmed and regime_filter:
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals