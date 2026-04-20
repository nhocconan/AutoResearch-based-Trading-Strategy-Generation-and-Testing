# 4h_1d_Donchian20_Volume_Confirm_Trend_v1
# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
# Works in bull markets by capturing breakouts and in bear markets via short breakdowns.
# Volume filter reduces false breakouts. Trend filter ensures directional alignment.
# Target: 20-50 trades/year (80-200 total over 4 years) to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(50) for long-term trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 4h volume filter (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_ratio_4h = vol_ratio[i]
        
        # Trend filter: price above EMA50 for uptrend, below for downtrend
        trend_up = price > ema_trend
        trend_down = price < ema_trend
        
        # Volume filter: require above-average volume
        vol_filter = vol_ratio_4h > 1.5
        
        if position == 0:
            # Enter long on Donchian breakout with trend and volume
            if price > highest_high[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian breakdown with trend and volume
            elif price < lowest_low[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian breakdown or trend reversal
            if price < lowest_low[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian breakout or trend reversal
            if price > highest_high[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Donchian20_Volume_Confirm_Trend_v1"
timeframe = "4h"
leverage = 1.0