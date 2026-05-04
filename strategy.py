#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# In trending markets (price > EMA50), we trade breakouts: long on upper Donchian breakout, short on lower.
# In ranging markets (price near EMA50), we fade Donchian extremes: short near upper band, long near lower band.
# Volume confirmation (>1.5x 20-period EMA) reduces false breakouts. Designed for 4h timeframe targeting 75-200 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "4h_Donchian20_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Determine market regime based on price vs EMA50
            if close[i] > ema_50_aligned[i]:
                # Uptrend: trade Donchian breakouts
                if close[i] > upper[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
            elif close[i] < ema_50_aligned[i]:
                # Downtrend: trade Donchian breakouts
                if close[i] < lower[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging market (price near EMA50): fade Donchian extremes
                if close[i] <= lower[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= upper[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches EMA50 OR volume drops
            if (close[i] <= ema_50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches EMA50 OR volume drops
            if (close[i] >= ema_50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals