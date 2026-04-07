#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v3
Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian upper band with above-average volume and 1d EMA50 trend bullish, enter short when price breaks below 20-period Donchian lower band with above-average volume and 1d EMA50 trend bearish. Exit when price crosses the 20-period EMA (mean reversion signal). Uses volume confirmation and trend filter to avoid false breakouts. Designed for 20-40 trades/year to minimize fee drag while capturing true momentum breaks in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 40-period Donchian channels (20-period for breakout, 40 for filter)
    donchian_window = 20
    if len(high) < donchian_window:
        return np.zeros(n)
    
    # Donchian upper and lower bands
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 20-period EMA for exit signal
    ema_period = 20
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_window, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 20-period EMA (mean reversion)
            if close[i] < ema[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 20-period EMA (mean reversion)
            if close[i] > ema[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian upper band with bullish 1d trend
                if high[i] > donchian_high[i] and ema50_1d_aligned[i] > close_1d[-1] if len(close_1d) > 0 else False:
                    # Simplified: use current 1d EMA50 > previous 1d close as trend proxy
                    if i > 0 and ema50_1d_aligned[i] > ema50_1d_aligned[i-1]:
                        position = 1
                        signals[i] = 0.25
                # Short: price breaks below Donchian lower band with bearish 1d trend
                elif low[i] < donchian_low[i]:
                    if i > 0 and ema50_1d_aligned[i] < ema50_1d_aligned[i-1]:
                        position = -1
                        signals[i] = -0.25
    
    return signals