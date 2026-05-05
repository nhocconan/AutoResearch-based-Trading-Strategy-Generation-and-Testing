#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian upper (20) AND 12h EMA50 uptrend AND volume spike
# Short when price breaks below Donchian lower (20) AND 12h EMA50 downtrend AND volume spike
# Uses Donchian channels for structure, 12h EMA50 for trend filter (avoid counter-trend),
# volume spike for conviction. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation).
# Timeframe: 4h (proven timeframe for balance of signal quality and trade frequency).

name = "4h_Donchian20_12hEMA50_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period) on 4h
    if len(high) >= 20:
        # Upper channel: highest high of last 20 periods
        upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Lower channel: lowest low of last 20 periods
        lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Higher threshold for fewer trades
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to insufficient data for indicators)
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel AND 12h EMA50 uptrend AND volume spike
            if (close[i] > upper[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND 12h EMA50 downtrend AND volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below upper channel OR closes below 12h EMA50
            if close[i] < upper[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above lower channel OR closes above 12h EMA50
            if close[i] > lower[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals