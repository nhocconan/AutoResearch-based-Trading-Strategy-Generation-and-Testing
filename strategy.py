#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d EMA34 for trend direction and 1d ATR-based volatility filter
# Entry: Price breaks above/below 12h Donchian(20) channel with volume confirmation (1.5x 20-period volume EMA)
# Trend filter: Only long when price > 1d EMA34, only short when price < 1d EMA34
# Exit: Opposite Donchian break OR price crosses 1d EMA34 (trend change)
# Uses discrete position sizing (0.25) to minimize fee churn
# Designed for low trade frequency (target: 12-37 trades/year) to overcome fee drag in ranging/bear markets
# Volume confirmation ensures strong participation, reducing false breakouts
# Works in both bull and bear markets by following 1d trend and using Donchian for structure

name = "12h_Donchian20_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < lookback - 1:
            continue
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Donchian breakout with 1d trend filter
        # Long: Price breaks above 12h Donchian upper + volume spike + price above 1d EMA34 (uptrend)
        # Short: Price breaks below 12h Donchian lower + volume spike + price below 1d EMA34 (downtrend)
        if position == 0:
            if (close[i] > highest_high[i] and volume_spike and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < lowest_low[i] and volume_spike and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below 12h Donchian lower OR price below 1d EMA34 (trend change)
            if close[i] < lowest_low[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above 12h Donchian upper OR price above 1d EMA34 (trend change)
            if close[i] > highest_high[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals