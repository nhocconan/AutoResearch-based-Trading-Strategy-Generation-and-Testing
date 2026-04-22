#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# Donchian breakouts capture trend momentum with clear entry/exit levels.
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Volume > 1.5x average confirms breakout strength.
# Works in both bull and bear markets by following the trend direction.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 trades over 4 years (12-37/year).

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
    
    # Donchian(20) channels on 12h data
    period = 20
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(period, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper band + above 1d EMA + volume spike
            if close[i] > upper[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band + below 1d EMA + volume spike
            elif close[i] < lower[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 1d EMA (trend change)
            if position == 1:
                # Exit long: Price below 1d EMA
                if close[i] < ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price above 1d EMA
                if close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0