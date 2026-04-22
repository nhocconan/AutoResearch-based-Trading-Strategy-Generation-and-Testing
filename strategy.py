#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d EMA(34) trend filter + volume confirmation
# Choppiness Index identifies ranging vs trending markets (range: >61.8, trend: <38.2)
# In trending regimes, we follow 1d EMA(34) direction with breakout entries
# Volume confirmation (>1.5x 20-period avg) filters false signals
# Designed for 4h timeframe targeting 20-40 trades/year with low churn

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Choppiness Index (14-period) on 4h data
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_num = np.log(atr_14 * 14) / np.log(2)
    chop_den = np.log(highest_high_14 - lowest_low_14)
    chop = 100 * chop_num / chop_den
    
    # Donchian Channel (20) for breakout entries
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop[i]) or np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in trending markets (Choppiness < 38.2)
        is_trending = chop[i] < 38.2
        
        if position == 0 and is_trending:
            # Long: Uptrend (price > 1d EMA) + breakout above upper Donchian + volume
            if (close[i] > ema_34_1d_aligned[i] and
                close[i] > highest_20[i-1] and
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < 1d EMA) + breakout below lower Donchian + volume
            elif (close[i] < ema_34_1d_aligned[i] and
                  close[i] < lowest_20[i-1] and
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or chop increases (range market)
            if position == 1:
                if (close[i] < ema_34_1d_aligned[i] or chop[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (close[i] > ema_34_1d_aligned[i] or chop[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Chop_Regime_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0