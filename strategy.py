# 4H_Donchian_Breakout_Volume_Trend_12hEMA50
# Hypothesis: Donchian(20) breakout on 4h with 12h EMA50 trend filter and volume spike confirmation.
# Uses tight entry conditions for low trade frequency (target: 20-50 trades/year) to avoid fee drag.
# Long when price breaks above Donchian upper channel + 12h EMA50 rising + volume spike.
# Short when price breaks below Donchian lower channel + 12h EMA50 falling + volume spike.
# Exits when price returns to Donchian middle or opposite breakout occurs.
# Designed to work in both bull and bear markets by following 12h trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 4h
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above upper channel + 12h EMA50 rising + volume spike
            if close[i] > upper[i] and ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel + 12h EMA50 falling + volume spike
            elif close[i] < lower[i] and ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle channel or opposite breakout
            exit_signal = False
            
            if position == 1:
                # Exit long: price closes below middle or breaks below lower channel
                if close[i] < middle[i] or close[i] < lower[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price closes above middle or breaks above upper channel
                if close[i] > middle[i] or close[i] > upper[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_Volume_Trend_12hEMA50"
timeframe = "4h"
leverage = 1.0