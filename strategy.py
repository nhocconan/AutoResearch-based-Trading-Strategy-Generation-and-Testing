#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(10) breakout with weekly EMA(50) trend filter and volume confirmation
# Donchian breakout captures momentum in both bull and bear markets.
# Weekly EMA(50) filter ensures we only trade in the direction of the long-term trend.
# Volume confirmation (>1.5x 12-period average) filters false breakouts.
# Designed for 12h timeframe targeting 15-25 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian Channel (10) on 12h data
    highest_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: 12-period average
    vol_avg_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_10[i]) or
            np.isnan(lowest_10[i]) or np.isnan(vol_avg_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band + weekly uptrend + volume confirmation
            if (close[i] > highest_10[i-1] and  # breakout above previous period's high
                close[i] > ema_50_1w_aligned[i] and  # price above weekly EMA (uptrend)
                volume[i] > 1.5 * vol_avg_12[i]):   # volume spike
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band + weekly downtrend + volume confirmation
            elif (close[i] < lowest_10[i-1] and   # breakout below previous period's low
                  close[i] < ema_50_1w_aligned[i] and  # price below weekly EMA (downtrend)
                  volume[i] > 1.5 * vol_avg_12[i]):   # volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band or trend reversal
            if position == 1:
                # Exit long: price returns to lower Donchian band or trend turns down
                if (close[i] < lowest_10[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns to upper Donchian band or trend turns up
                if (close[i] > highest_10[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian10_WeeklyEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0