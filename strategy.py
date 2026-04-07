#!/usr/bin/env python3
"""
1d_donchian_20_1w_trend_volume_v2
Hypothesis: On daily timeframe, use weekly Donchian channels (20-week high/low) for trend-following breakouts, filtered by weekly EMA trend and daily volume confirmation. This strategy captures major trend moves while avoiding whipsaws in ranging markets. Designed for 30-100 total trades over 4 years (~7-25/year) to minimize fee drift while performing in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_1w_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and EMA trend
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA(20) to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: max of last 20 weekly highs
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: min of last 20 weekly lows
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume filter: 20-day average on daily timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 30), n):
        # Skip if data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly Donchian low (trend reversal)
            if low[i] <= donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly Donchian high (trend reversal)
            if high[i] >= donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout entry: price breaks above/below weekly Donchian channels with trend alignment
                # Long: break above Donchian high with upward EMA trend
                if high[i] >= donchian_high_aligned[i] and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                    position = 1
                    signals[i] = 0.30
                # Short: break below Donchian low with downward EMA trend
                elif low[i] <= donchian_low_aligned[i] and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                    position = -1
                    signals[i] = -0.30
    
    return signals