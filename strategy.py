#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and 1d volume confirmation.
# Breakouts in the direction of the daily trend capture momentum while avoiding counter-trend whipsaws.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# Designed for 6h timeframe targeting 15-30 trades/year per symbol.
# Works in bull markets by capturing upward breakouts and in bear markets by avoiding false breakdowns
# via trend filter and requiring volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and volume confirmation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume 20-period average for spike detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Donchian Channel (20-period) on 6h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band + 1d uptrend + volume spike
            if (close[i] > highest_high_20[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band + 1d downtrend + volume spike
            elif (close[i] < lowest_low_20[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to midpoint of Donchian channel or trend reversal
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2.0
            if position == 1:
                # Exit long on return to midpoint or trend reversal
                if (close[i] <= midpoint or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short on return to midpoint or trend reversal
                if (close[i] >= midpoint or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0