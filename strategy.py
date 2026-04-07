#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) Breakout + 12h EMA Trend + Volume Confirmation
# Hypothesis: Breakouts in direction of 12h EMA(40) trend with volume > 20-period average capture strong moves in both bull/bear markets.
# Uses discrete position sizing (0.25) to limit trade frequency and reduce fee drag. Target: 20-50 trades/year.

name = "4h_donchian20_volume_12h_trend_v1"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate EMA(40) on 12h close
    close_12h = df_12h['close'].values
    ema_40_12h = pd.Series(close_12h).ewm(span=40, adjust=False).mean().values
    ema_40_4h = align_htf_to_ltf(prices, df_12h, ema_40_12h)
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 4h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_40_4h[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend changes
            if close[i] < low_20[i] or close[i] < ema_40_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend changes
            if close[i] > high_20[i] or close[i] > ema_40_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of 12h EMA trend with volume confirmation
            if vol_ok:
                if close[i] > ema_40_4h[i]:  # Uptrend
                    if high[i] > high_20[i]:  # Break above Donchian high
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if low[i] < low_20[i]:  # Break below Donchian low
                        position = -1
                        signals[i] = -0.25
    
    return signals