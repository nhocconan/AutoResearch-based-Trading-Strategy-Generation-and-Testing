#!/usr/bin/env python3
# 4h_donchian_breakout_1d_vol_chop_v3
# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter.
# Works in bull/bear: Donchian captures breakouts; 1d volume > 1.5x 20d average confirms institutional participation;
# Choppiness index (14) > 61.8 = range (mean revert), < 38.2 = trending (trend follow) on 1d.
# In trending regime (CHOP < 38.2): follow breakout direction. In range regime (CHOP > 61.8): fade breakouts.
# Target: 20-50 trades/year, discrete size 0.25.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_vol_chop_v3"
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
    
    # 1d HTF data for volume, chop, and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for indicators
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: current volume > 1.5x 20-period average
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = align_htf_to_ltf(prices, df_1d, volume_1d > 1.5 * volume_ma_1d)
    
    # 1d Choppiness Index (14) - measures if market is ranging or trending
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high_n) - min(low_n))))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First period
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    max_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14_1d = max_high_14_1d - min_low_14_1d
    chop_1d = np.where(range_14_1d > 0, 100 * np.log10(pd.Series(atr_14_1d).rolling(14, min_periods=14).sum().values / (np.log10(14) * range_14_1d)), 50)
    chop_1d = np.where(np.isnan(chop_1d), 50, chop_1d)  # Default to neutral if calculation fails
    
    chop_align = align_htf_to_ltf(prices, df_1d, chop_1d)
    chop_trending = chop_align < 38.2  # Trending regime
    chop_ranging = chop_align > 61.8   # Ranging regime
    
    # 4h Donchian channels (20-period)
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_window, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_confirmed_1d[i]) or
            np.isnan(chop_align[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR chop becomes ranging (mean reversion opportunity)
            if close[i] < lowest_low[i] or chop_ranging[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR chop becomes ranging (mean reversion opportunity)
            if close[i] > highest_high[i] or chop_ranging[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            if not volume_confirmed_1d[i]:
                signals[i] = 0.0
                continue
                
            # In trending regime: follow breakout
            if chop_trending[i]:
                # Long: price breaks above Donchian high
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
            # In ranging regime: fade breakouts (mean reversion)
            elif chop_ranging[i]:
                # Long: price breaks below Donchian low (oversold bounce)
                if close[i] < lowest_low[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks above Donchian high (overbought reversal)
                elif close[i] > highest_high[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals