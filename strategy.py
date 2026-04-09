#!/usr/bin/env python3
# 4h_volume_spike_donchian_breakout_chop_regime_v1
# Hypothesis: 4h Donchian(20) breakout with volume spike (>2x average) and choppy market regime (CHOP > 61.8) for mean reversion.
# In choppy markets (CHOP > 61.8), fade the breakout; in trending markets (CHOP <= 61.8), follow the breakout.
# Uses 1d HTF for trend filter: only take longs when price > 1d EMA(50), shorts when price < 1d EMA(50).
# Designed to work in both bull and bear markets via regime adaptation and HTF trend filter.
# Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volume_spike_donchian_breakout_chop_regime_v1"
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
    
    # 1d HTF data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP) - 14-period
    def calculate_chop(high, low, close, window=14):
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=window, min_periods=window).sum()
        highest_high_window = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low_window = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr / (highest_high_window - lowest_low_window)) / np.log10(window)
        return chop.values
    
    chop = calculate_chop(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(chop[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish (price < 1d EMA)
            if close[i] < lowest_low[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish (price > 1d EMA)
            if close[i] > highest_high[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Regime filter: CHOP > 61.8 = choppy (mean revert), CHOP <= 61.8 = trending (follow breakout)
                is_choppy = chop[i] > 61.8
                
                # Breakout detection
                breakout_up = close[i] > highest_high[i]
                breakout_down = close[i] < lowest_low[i]
                
                if is_choppy:
                    # In choppy markets: fade the breakout (mean reversion)
                    if breakout_up and close[i] < ema_1d_aligned[i]:  # Failed breakout above resistance
                        position = -1
                        signals[i] = -0.25
                    elif breakout_down and close[i] > ema_1d_aligned[i]:  # Failed breakout below support
                        position = 1
                        signals[i] = 0.25
                else:
                    # In trending markets: follow the breakout with HTF trend filter
                    if breakout_up and close[i] > ema_1d_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    elif breakout_down and close[i] < ema_1d_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals