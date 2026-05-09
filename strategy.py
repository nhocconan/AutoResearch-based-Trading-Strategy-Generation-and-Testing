#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation
# Uses Donchian(20) breakouts for trend following, 12h EMA50 for trend filter,
# and volume spike for confirmation. Designed for 20-50 trades/year to avoid
# fee drag. Works in bull markets (breakouts with trend) and bear markets
# (avoids counter-trend trades via EMA filter).
name = "4h_DonchianBreakout_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above upper Donchian with 12h uptrend and volume spike
            if close[i] > high_20[i] and close[i] > ema50_4h[i] and vol_spike:
                signals[i] = 0.30
                position = 1
            # Short: Break below lower Donchian with 12h downtrend and volume spike
            elif close[i] < low_20[i] and close[i] < ema50_4h[i] and vol_spike:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below lower Donchian OR 12h trend turns down
            if close[i] < low_20[i] or close[i] < ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: Price rises back above upper Donchian OR 12h trend turns up
            if close[i] > high_20[i] or close[i] > ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals