#!/usr/bin/env python3
# mtf_1h_donchian_breakout_volume_chop_4h1d_v1
# Hypothesis: 1h strategy using 4h/1d Donchian(20) breakouts with volume confirmation and chop regime filter.
# Uses 4h for trend direction (price > 4h Donchian mid = long bias, < mid = short bias),
# 1d for chop regime filter (only trade when chop > 50 = ranging market),
# 1h for entry timing with volume confirmation.
# In ranging markets (2025+), price tends to revert from Donchian channel extremes.
# Volume confirmation filters false breakouts. Chop filter ensures favorable conditions.
# Discrete sizing (0.0, ±0.20) minimizes fee churn. Target: 15-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_breakout_volume_chop_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for trend direction (Donchian mid)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (high_20_4h + low_20_4h) / 2.0
    donchian_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # 1d HTF data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_14_1d = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum().values
    
    # Calculate chop: chop > 50 = ranging market (favorable for mean reversion)
    chop_denom = np.log10(atr_14_1d) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14_1d - low_14_1d) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN or outside session
        if (np.isnan(high_20_4h[i]) or np.isnan(low_20_4h[i]) or 
            np.isnan(donchian_mid_4h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging (chop > 50)
        chop_regime = chop_aligned[i] > 50
        
        if position == 1:  # Long position
            # Exit: price moves below 4h Donchian low or volume dries up or chop breaks down
            if (close[i] < low_20_4h[i] or not volume_confirmed or 
                chop_aligned[i] < 40):  # Exit if chop < 40 (trending)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price moves above 4h Donchian high or volume dries up or chop breaks down
            if (close[i] > high_20_4h[i] or not volume_confirmed or 
                chop_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price above 4h Donchian mid (bullish bias) + breaks above 1h Donchian high
                # Calculate 1h Donchian for entry timing
                high_20_1h = pd.Series(high[:i+1]).rolling(window=20, min_periods=20).max().iloc[-1]
                low_20_1h = pd.Series(low[:i+1]).rolling(window=20, min_periods=20).min().iloc[-1]
                
                if (close[i] > donchian_mid_4h_aligned[i] and  # 4h bullish bias
                    close[i] > high_20_1h):  # 1h breakout confirmation
                    position = 1
                    signals[i] = 0.20
                # Short entry: price below 4h Donchian mid (bearish bias) + breaks below 1h Donchian low
                elif (close[i] < donchian_mid_4h_aligned[i] and  # 4h bearish bias
                      close[i] < low_20_1h):  # 1h breakdown confirmation
                    position = -1
                    signals[i] = -0.20
    
    return signals