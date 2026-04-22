#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Uses 6h price channel breakouts, 12h EMA for trend direction, and volume spike for confirmation.
# Long when price breaks above Donchian high in uptrend with volume spike.
# Short when price breaks below Donchian low in downtrend with volume spike.
# Exit on opposite Donchian touch or trend reversal.
# Designed for 6h timeframe to target 15-35 trades/year per symbol.
# Works in bull/bear via trend filter + volatility-based entry levels.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(34) for trend direction
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 6h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + uptrend + volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + downtrend + volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on Donchian low touch or trend reversal
                if (close[i] < low_min[i] or close[i] < ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Donchian high touch or trend reversal
                if (close[i] > high_max[i] or close[i] > ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0