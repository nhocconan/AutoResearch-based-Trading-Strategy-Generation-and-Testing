#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + Daily Trend + Volume Confirmation
# Hypothesis: Donchian(20) breakouts capture momentum. Daily trend (EMA50 > SMA50) filters direction.
# Volume confirms institutional participation. Works in bull (breakouts continue) and bear (breaks down).
# 4h timeframe balances responsiveness and noise. Target: 20-50 trades/year (80-200 over 4 years).
name = "4h_donchian_breakout_daily_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter: EMA50 > SMA50 = bullish
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_sma50 = pd.Series(daily_close).rolling(window=50, min_periods=50).mean().values
    daily_ema50_4h = align_htf_to_ltf(prices, df_1d, daily_ema50)
    daily_sma50_4h = align_htf_to_ltf(prices, df_1d, daily_sma50)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(daily_ema50_4h[i]) or np.isnan(daily_sma50_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime from daily data
        bull_regime = daily_ema50_4h[i] > daily_sma50_4h[i]  # Bullish when EMA > SMA
        bear_regime = daily_ema50_4h[i] < daily_sma50_4h[i]  # Bearish when EMA < SMA
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian channel or trend reversal
            if close[i] < low_20[i] or not bull_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian channel or trend reversal
            if close[i] > high_20[i] or not bear_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Bull regime: look for long when price breaks above upper Donchian
                if bull_regime and close[i] > high_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Bear regime: look for short when price breaks below lower Donchian
                elif bear_regime and close[i] < low_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals