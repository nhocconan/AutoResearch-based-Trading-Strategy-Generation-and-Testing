#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# Donchian breakouts capture strong momentum moves in both bull and bear markets
# 1d volume spike (>2.0 x 20-period EMA) confirms institutional participation
# Choppiness regime (CHOP > 61.8) filters for ranging markets where mean reversion works better
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_Donchian20_Breakout_1dVolume_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d data for volume and choppiness filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ema_20_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1d choppiness regime filter
    # CHOP > 61.8 = ranging market (favor mean reversion)
    # CHOP < 38.2 = trending market (favor trend following)
    # We'll use CHOP > 50 as a balanced filter for both regimes
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    atr_safe = np.where(atr_14_1d == 0, 1e-10, atr_14_1d)
    chop_1d = 100 * np.log10((highest_high_14_1d - lowest_low_14_1d) / (np.sum(pd.Series(tr).rolling(window=14, min_periods=14).sum().values) / 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 34)  # enough for Donchian and 1d indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime
        is_ranging = chop_aligned[i] > 50  # CHOP > 50 = ranging/mean reverting
        is_trending = chop_aligned[i] <= 50  # CHOP <= 50 = trending
        
        if position == 0:  # Flat - look for new entries
            # In ranging markets: look for mean reversion at Donchian bands
            # In trending markets: look for breakouts
            
            if is_ranging:
                # Mean reversion: buy near lower band, sell near upper band
                if close[i] <= lowest_low[i] and volume_spike_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= highest_high[i] and volume_spike_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # trending
                # Breakout: buy on upper band break, sell on lower band break
                if close[i] > highest_high[i] and volume_spike_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low[i] and volume_spike_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price reaches opposite Donchian band or loss of momentum
            if close[i] >= highest_high[i] or (not volume_spike_aligned[i] and chop_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price reaches opposite Donchian band or loss of momentum
            if close[i] <= lowest_low[i] or (not volume_spike_aligned[i] and chop_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals