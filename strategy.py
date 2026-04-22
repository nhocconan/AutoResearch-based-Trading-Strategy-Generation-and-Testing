#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation.
# Uses 1w EMA(50) as primary trend filter on weekly timeframe to capture long-term direction.
# Long when price breaks above 20-day high with bullish weekly trend and volume spike.
# Short when price breaks below 20-day low with bearish weekly trend and volume spike.
# Designed for 1d timeframe to target 15-25 trades/year per symbol with low frequency and high win rate.
# Works in bull/bear via weekly trend alignment and volatility-based breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channel calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    # Upper band = highest high over last 20 days
    # Lower band = lowest low over last 20 days
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Load 1w data for EMA(50) trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for higher timeframe trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike filter (20-period on 1d data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer, higher quality trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high + bullish weekly trend + volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low + bearish weekly trend + volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on break below 20-day low or trend reversal
                if (close[i] < low_20[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on break above 20-day high or trend reversal
                if (close[i] > high_20[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0