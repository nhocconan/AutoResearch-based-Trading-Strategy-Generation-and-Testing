#!/usr/bin/env python3
# 1d_donchian_1w_trend_volume_v1
# Hypothesis: Daily Donchian(20) breakout with weekly EMA(50) trend filter and volume confirmation.
# Long: price breaks above 20-day high + price > weekly EMA50 + volume > 1.5x 20-day avg volume
# Short: price breaks below 20-day low + price < weekly EMA50 + volume > 1.5x 20-day avg volume
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 50-100 total trades over 4 years.
# Works in bull markets via breakouts and bear markets via short breakdowns with trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels (20-period) on daily data
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 20-day low OR weekly trend turns bearish
            if close[i] < period20_low[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 20-day high OR weekly trend turns bullish
            if close[i] > period20_high[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above 20-day high + above weekly EMA50
                if close[i] > period20_high[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below 20-day low + below weekly EMA50
                elif close[i] < period20_low[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals