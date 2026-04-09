#!/usr/bin/env python3
# 1d_donchian_breakout_volume_chop_regime_v1
# Hypothesis: Daily Donchian(20) breakout with volume confirmation and choppiness regime filter.
# In ranging markets (2025+), price tends to revert from Donchian channel extremes.
# Volume confirmation filters false breakouts. Choppiness filter ensures we only trade in ranging regimes.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 30-100 total trades over 4 years.
# Primary timeframe: 1d, HTF: 1w for higher-timeframe trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_chop_regime_v1"
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
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channel (20-period) on daily data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness index regime filter (14-period)
    high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging (chop between 30 and 70)
        chop_regime = (chop[i] > 30) & (chop[i] < 70)
        
        # Trend filter: only trade long when price > weekly EMA50, short when price < weekly EMA50
        trend_filter_long = close[i] > ema_50_1w_aligned[i]
        trend_filter_short = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian lower band or volume dries up
            if close[i] < low_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian upper band or volume dries up
            if close[i] > high_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price breaks above Donchian upper band with volume confirmation
                if close[i] > high_20[i] and trend_filter_long:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian lower band with volume confirmation
                elif close[i] < low_20[i] and trend_filter_short:
                    position = -1
                    signals[i] = -0.25
    
    return signals