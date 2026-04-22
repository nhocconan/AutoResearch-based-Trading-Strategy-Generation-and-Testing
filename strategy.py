#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA34 trend filter and volume confirmation
# Donchian channel breakouts capture momentum moves in both bull and bear markets.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume > 1.5x 20-period average confirms breakout strength.
# Exit when price returns to the middle of the Donchian channel.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 20-50 trades/year per symbol to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channel (20-period)
    donchian_len = 20
    upper = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    middle = (upper + lower) / 2
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(donchian_len, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper Donchian + above 1d EMA + volume spike
            if close[i] > upper[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price < lower Donchian + below 1d EMA + volume spike
            elif close[i] < lower[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle of Donchian channel
            if position == 1:
                if close[i] <= middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0