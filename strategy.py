#!/usr/bin/env python3
# 1d_Weekly_Trend_Filtered_Breakout_v2
# Hypothesis: Capture weekly trend continuation with daily breakouts. In bull/bear markets, price tends to continue in the direction of the weekly trend after consolidating. Enter on daily breakouts in the direction of the weekly trend with volume confirmation. Exit when weekly trend reverses or volatility expands. Designed for low trade frequency (7-25/year) to minimize fee drag on 1d timeframe.

name = "1d_Weekly_Trend_Filtered_Breakout_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend: EMA21/EMA50 crossover
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w = np.where(ema21_1w > ema50_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volatility filter: ATR(20) < ATR(50) to avoid high volatility periods
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    low_vol = atr20 < atr50  # Lower recent volatility
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(atr20[i]) or np.isnan(atr50[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: weekly trend turns bearish OR price breaks below Donchian low
            if trend_1w_aligned[i] == -1 or low[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly trend turns bullish OR price breaks above Donchian high
            if trend_1w_aligned[i] == 1 or high[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter in direction of weekly trend with confirmation
            if trend_1w_aligned[i] == 1 and high[i] > high_20[i]:
                # Weekly uptrend + break above Donchian high + low volatility + volume
                if low_vol[i] and vol_confirm[i]:
                    position = 1
                    signals[i] = 0.25
            elif trend_1w_aligned[i] == -1 and low[i] < low_20[i]:
                # Weekly downtrend + break below Donchian low + low volatility + volume
                if low_vol[i] and vol_confirm[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals