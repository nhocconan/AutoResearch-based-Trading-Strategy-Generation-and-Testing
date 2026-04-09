#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and chop regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trending).
# In trending regime (CHOP < 38.2): breakout long above upper band, short below lower band.
# In range regime (CHOP > 61.8): mean reversion - short at upper band, long at lower band.
# Exit when price closes back inside Donchian bands.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 20-40 trades/year (80-160 total over 4 years) on BTC/ETH/SOL.
# Works in bull/bear via regime adaptation: trend follow in trending markets, mean revert in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v3"
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
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Chopiness Index (14-period) for regime detection
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    
    atr_series = pd.Series(tr)
    atr = atr_series.rolling(window=14, min_periods=14).mean().values
    
    max_high = high_series.rolling(window=14, min_periods=14).max().values
    min_low = low_series.rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (max_high - min_low)) / log10(14)
    sum_tr = atr_series.rolling(window=14, min_periods=14).sum().values
    denominator = max_high - min_low
    chop = np.where(denominator > 0, 100 * np.log10(sum_tr / denominator) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price closes back below Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes back above Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Regime-based entry logic
            if chop[i] < 38.2:  # Trending regime - trend follow
                bullish_breakout = (close[i] > donchian_high[i]) and volume_confirmed
                bearish_breakout = (close[i] < donchian_low[i]) and volume_confirmed
                
                if bullish_breakout:
                    position = 1
                    signals[i] = 0.25
                elif bearish_breakout:
                    position = -1
                    signals[i] = -0.25
                    
            elif chop[i] > 61.8:  # Range regime - mean revert
                bullish_bounce = (close[i] < donchian_low[i]) and volume_confirmed
                bearish_bounce = (close[i] > donchian_high[i]) and volume_confirmed
                
                if bullish_bounce:
                    position = 1
                    signals[i] = 0.25
                elif bearish_bounce:
                    position = -1
                    signals[i] = -0.25
    
    return signals