#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA(20) trend filter and volume confirmation.
# Uses daily price channels for structure, weekly EMA for trend direction, and volume spike for confirmation.
# Long when price breaks above upper Donchian in uptrend (close > weekly EMA20) with volume spike.
# Short when price breaks below lower Donchian in downtrend (close < weekly EMA20) with volume spike.
# Exit on opposite Donchian touch or trend reversal.
# Designed for 1d timeframe to target 7-25 trades/year per symbol.
# Works in bull/bear via trend filter + volatility-based entry levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels for 20-day period
    # Upper = max(high, 20)
    # Lower = min(low, 20)
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 20-period EMA on weekly close for trend direction
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 1d timeframe (waits for daily bar to close)
    upper_donchian = align_htf_to_ltf(prices, df_1d, high_max)
    lower_donchian = align_htf_to_ltf(prices, df_1d, low_min)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1w)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + uptrend (close > weekly EMA20) + volume spike
            if (close[i] > upper_donchian[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + downtrend (close < weekly EMA20) + volume spike
            elif (close[i] < lower_donchian[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on lower Donchian touch or trend reversal
                if (close[i] < lower_donchian[i] or close[i] < ema_20_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on upper Donchian touch or trend reversal
                if (close[i] > upper_donchian[i] or close[i] > ema_20_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA20_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0