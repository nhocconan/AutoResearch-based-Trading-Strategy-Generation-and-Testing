#!/usr/bin/env python3
# 1d_donchian_volume_chop_regime_v1
# Hypothesis: 1d Donchian(20) breakout with volume confirmation (>1.3x 20-day average) and choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend).
# In trending regimes (CHOP < 38.2): follow Donchian breakout direction.
# In ranging regimes (CHOP > 61.8): mean revert at Donchian channels (sell at upper, buy at lower).
# Weekly trend filter (price above/below weekly 20 EMA) avoids counter-trend trades.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_volume_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_w = df_1w['close'].values
    
    # Weekly 20 EMA for trend filter
    ema_20_1w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Donchian channels (20-day)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-day)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-day)
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.log10(hh - ll)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or np.isnan(ema_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-day average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian upper or volume dries up
            if close[i] >= donchian_upper[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Donchian lower or volume dries up
            if close[i] <= donchian_lower[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Determine regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
                if chop[i] < 38.2:  # Trending regime - follow breakout
                    # Long breakout: price breaks above Donchian upper AND weekly trend filter (price > weekly 20 EMA)
                    if close[i] > donchian_upper[i] and close[i] > ema_20_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short breakdown: price breaks below Donchian lower AND weekly trend filter (price < weekly 20 EMA)
                    elif close[i] < donchian_lower[i] and close[i] < ema_20_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                elif chop[i] > 61.8:  # Ranging regime - mean revert at channels
                    # Short at upper channel: price touches Donchian upper AND weekly trend filter (price < weekly 20 EMA for short bias)
                    if close[i] >= donchian_upper[i] and close[i] < ema_20_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                    # Long at lower channel: price touches Donchian lower AND weekly trend filter (price > weekly 20 EMA for long bias)
                    elif close[i] <= donchian_lower[i] and close[i] > ema_20_aligned[i]:
                        position = 1
                        signals[i] = 0.25
    
    return signals