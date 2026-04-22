#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume spike confirmation.
# Uses 12h Donchian channels for structure, 12h EMA for trend direction, and volume spike for confirmation.
# Long when price breaks above 12h upper channel in uptrend (close > 12h EMA50) with volume spike.
# Short when price breaks below 12h lower channel in downtrend (close < 12h EMA50) with volume spike.
# Exit on opposite channel touch or trend reversal.
# Designed for 4h timeframe to target 20-50 trades/year per symbol.
# Works in bull/bear via trend filter + volatility-based entry levels.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian and trend (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_max = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h timeframe (waits for 12h bar to close)
    upper_channel = align_htf_to_ltf(prices, df_12h, high_max)
    lower_channel = align_htf_to_ltf(prices, df_12h, low_min)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel + uptrend (close > EMA50) + volume spike
            if (close[i] > upper_channel[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel + downtrend (close < EMA50) + volume spike
            elif (close[i] < lower_channel[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on lower channel touch or trend reversal
                if (close[i] < lower_channel[i] or close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on upper channel touch or trend reversal
                if (close[i] > upper_channel[i] or close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0