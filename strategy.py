#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above 20-period Donchian high with volume > 1.5x average and CHOP > 61.8 (ranging market).
# Short when price breaks below 20-period Donchian low with volume > 1.5x average and CHOP > 61.8.
# Exit when price closes back inside the Donchian channel (opposite side) or CHOP < 38.2 (strong trend).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture mean-reversion breakouts in ranging markets while avoiding strong trends.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v1"
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
    
    # Donchian channel (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * max(high-low))) / log10(n)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # first TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    max_min_range = pd.Series(high - low).rolling(window=14, min_periods=14).max().values
    chop = 100 * (np.log10(atr1) - np.log10(14 * max_min_range)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: CHOP > 61.8 for ranging market (mean reversion)
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price closes back below Donchian low or CHOP < 38.2 (strong trend)
            if close[i] < donchian_low[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes back above Donchian high or CHOP < 38.2 (strong trend)
            if close[i] > donchian_high[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation and ranging market
            bullish_breakout = (close[i] > donchian_high[i]) and volume_confirmed and ranging_market
            bearish_breakout = (close[i] < donchian_low[i]) and volume_confirmed and ranging_market
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals