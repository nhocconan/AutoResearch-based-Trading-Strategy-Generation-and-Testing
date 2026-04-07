#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with 12h Volume and 1d Trend Filter
# Hypothesis: Donchian channel breakouts capture momentum moves; 12h volume confirms institutional participation; 1d EMA filter ensures alignment with higher timeframe trend.
# Works in bull via upper band breakouts, in bear via lower band breakdowns. Volume and trend filters reduce false breakouts.
# Target: 25-40 trades/year to minimize fee drag.
name = "4h_donchian20_12h_volume_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h volume moving average
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian Channel (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 12h average volume
        vol_confirm = volume[i] > vol_ma_12h_aligned[i]
        
        # Trend filter: price above/below 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low (mean reversion or trend change)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (mean reversion or trend change)
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian high + volume confirmation + uptrend
            if close[i] > donchian_high[i] and vol_confirm and above_ema:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian low + volume confirmation + downtrend
            elif close[i] < donchian_low[i] and vol_confirm and below_ema:
                position = -1
                signals[i] = -0.25
    
    return signals