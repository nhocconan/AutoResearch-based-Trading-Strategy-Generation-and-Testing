#!/usr/bin/env python3
# 6h_donchian_12h_trend_volume_v1
# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 12-37 trades/year.
# Works in bull/bear by aligning with 12h trend via EMA50. Volume confirms institutional participation.
# 6h timeframe avoids overtrading while capturing medium-term moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for Donchian channels and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h Donchian(20) channels
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below upper Donchian OR trend turns bearish
            if close[i] < upper_20_aligned[i] or close[i] < ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above lower Donchian OR trend turns bullish
            if close[i] > lower_20_aligned[i] or close[i] > ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.8 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above upper Donchian with bullish trend
                if close[i] > upper_20_aligned[i] and close[i] > ema50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian with bearish trend
                elif close[i] < lower_20_aligned[i] and close[i] < ema50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals