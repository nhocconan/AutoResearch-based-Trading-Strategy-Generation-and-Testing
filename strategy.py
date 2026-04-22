#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA(20) trend filter and volume confirmation.
# Uses 6h Donchian channels for breakout signals, 12h EMA for trend direction, and volume spike for confirmation.
# Long when price breaks above 6h Donchian upper channel in uptrend (close > 12h EMA20) with volume spike.
# Short when price breaks below 6h Donchian lower channel in downtrend (close < 12h EMA20) with volume spike.
# Exit on opposite Donchian touch or trend reversal.
# Designed for 6h timeframe to target 12-37 trades/year per symbol.
# Works in bull/bear via trend filter + volatility-based entry levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(20) for trend direction
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 6h timeframe (waits for 12h bar to close)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # 6h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + uptrend (close > 12h EMA20) + volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema_20_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + downtrend (close < 12h EMA20) + volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema_20_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on Donchian lower touch or trend reversal
                if (close[i] < low_20[i] or close[i] < ema_20_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Donchian upper touch or trend reversal
                if (close[i] > high_20[i] or close[i] > ema_20_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA20_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0