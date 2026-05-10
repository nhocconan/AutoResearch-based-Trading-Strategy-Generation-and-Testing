# 12h_Donchian_Breakout_Trend_With_Volume_Filter
# Hypothesis: Uses 12h Donchian(20) breakout with trend confirmation (price above/below 12h EMA50) and volume filter (volume > 1.5x average volume) to capture strong trends.
# Designed for low trade frequency (target: 20-50 trades/year) with strong trend persistence in both bull and bear markets.
# Uses daily (1d) EMA200 as higher timeframe trend filter to avoid counter-trend trades.

name = "12h_Donchian_Breakout_Trend_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # 1d EMA200 for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema50[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + above EMA50 + volume filter + 1d uptrend (price > EMA200)
            if (close[i] > donch_high[i] and close[i] > ema50[i] and 
                volume_filter[i] and close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below EMA50 + volume filter + 1d downtrend (price < EMA200)
            elif (close[i] < donch_low[i] and close[i] < ema50[i] and 
                  volume_filter[i] and close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below Donchian low or below EMA50 (trend weakness)
            if close[i] < donch_low[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above Donchian high or above EMA50 (trend weakness)
            if close[i] > donch_high[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals