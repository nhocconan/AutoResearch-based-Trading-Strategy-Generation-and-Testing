#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation
# Donchian breakout captures breakout momentum in both bull and bear markets.
# 12h EMA filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation (>1.5x 20-period average) filters false breakouts.
# Designed for 4h timeframe targeting 20-40 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian Channel (20) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band + 12h uptrend + volume confirmation
            if (close[i] > highest_20[i-1] and  # breakout above previous period's high
                close[i] > ema_50_12h_aligned[i] and  # price above 12h EMA (uptrend)
                volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band + 12h downtrend + volume confirmation
            elif (close[i] < lowest_20[i-1] and   # breakout below previous period's low
                  close[i] < ema_50_12h_aligned[i] and  # price below 12h EMA (downtrend)
                  volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band or trend reversal
            if position == 1:
                # Exit long: price returns to lower Donchian band or trend turns down
                if (close[i] < lowest_20[i] or 
                    close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns to upper Donchian band or trend turns up
                if (close[i] > highest_20[i] or 
                    close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0