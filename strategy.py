#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(25) breakout with 1d EMA(34) trend filter and volume confirmation
# Donchian breakout captures breakout momentum in both bull and bear markets.
# 1d EMA(34) filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation (>1.5x 20-period average) filters false breakouts.
# Designed for 4h timeframe targeting 20-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian Channel (25) on 4h data
    highest_25 = pd.Series(high).rolling(window=25, min_periods=25).max().values
    lowest_25 = pd.Series(low).rolling(window=25, min_periods=25).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_25[i]) or
            np.isnan(lowest_25[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band + 1d uptrend + volume confirmation
            if (close[i] > highest_25[i-1] and  # breakout above previous period's high
                close[i] > ema_34_1d_aligned[i] and  # price above 1d EMA (uptrend)
                volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band + 1d downtrend + volume confirmation
            elif (close[i] < lowest_25[i-1] and   # breakout below previous period's low
                  close[i] < ema_34_1d_aligned[i] and  # price below 1d EMA (downtrend)
                  volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band or trend reversal
            if position == 1:
                # Exit long: price returns to lower Donchian band or trend turns down
                if (close[i] < lowest_25[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns to upper Donchian band or trend turns up
                if (close[i] > highest_25[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian25_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0