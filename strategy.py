#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + 12h Trend + Volume Confirmation
# Hypothesis: Combines price breakouts from Donchian channels (20-period) with
# 12h EMA trend filter and volume confirmation to capture strong momentum moves.
# Works in bull markets via upside breakouts and in bear markets via downside breakdowns.
# Volume confirmation filters out false breakouts. Target: 20-40 trades/year.
name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # Donchian channels (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA(34) for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: 4h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or trend turns bearish
            if close[i] < low_20[i] or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or trend turns bullish
            if close[i] > high_20[i] or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian upper band with volume and bullish trend
            if close[i] > high_20[i] and vol_confirm and close[i] > ema_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian lower band with volume and bearish trend
            elif close[i] < low_20[i] and vol_confirm and close[i] < ema_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals