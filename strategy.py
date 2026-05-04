#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses daily Donchian channel breakouts for clear entry/exit signals with defined risk
# 1w EMA34 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (>1.8x 20-period EMA volume) filters false breakouts
# Discrete sizing 0.25 targets 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Works in bull markets (continuation breakouts with uptrend) and bear markets (continuation breakdowns with downtrend)
# 1w trend filter ensures alignment with longer-term structure, reducing whipsaws in ranging markets

name = "1d_Donchian20_1wEMA34_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34) trend filter from prior completed 1w bar
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_shifted = np.roll(ema_34_1w, 1)
    ema_34_1w_shifted[0] = np.nan
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels from prior completed 1d bar
    high_1d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_1d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Shift to use prior completed 1d bar (avoid look-ahead)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    high_1d_shifted[0] = np.nan
    low_1d_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_1d_shifted[i]) or 
            np.isnan(low_1d_shifted[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND price > 1w EMA34 AND volume spike
            if close[i] > high_1d_shifted[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND price < 1w EMA34 AND volume spike
            elif close[i] < low_1d_shifted[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian lower band OR price crosses below 1w EMA34
            if close[i] < low_1d_shifted[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian upper band OR price crosses above 1w EMA34
            if close[i] > high_1d_shifted[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals