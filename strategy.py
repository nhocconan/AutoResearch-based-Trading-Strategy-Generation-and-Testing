#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with 1d Trend and Volume Confirmation v1
# Hypothesis: Donchian(20) breakouts on 4h with 1d trend filter (EMA50) and volume spike
# capture strong momentum moves. Works in bull markets via breakouts and in bear markets
# via breakdowns. Volume confirmation reduces false breakouts. Target: 20-50 trades/year.

name = "4h_donchian_breakout_1d_trend_volume_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4-period volume average for spike detection
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=4, min_periods=1).mean().values
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=1).max().values
    donchian_low = low_series.rolling(window=20, min_periods=1).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] < donchian_low[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] > donchian_high[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Long: price breaks above Donchian high AND uptrend AND volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_1d_aligned[i] and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND downtrend AND volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_1d_aligned[i] and volume_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals