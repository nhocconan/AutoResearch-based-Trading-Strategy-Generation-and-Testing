#!/usr/bin/env python3
"""
1h_4h1d_EMA_Crossover_Signal
Hypothesis: Combines 4h EMA trend (EMA34) with 1d EMA trend filter (EMA50) and 1h EMA entry (EMA12/26).
Only takes long when 4h and 1d EMAs are bullish (price > EMA) and 1h EMA12 crosses above EMA26.
Short when 4h and 1d EMAs are bearish and 1h EMA12 crosses below EMA26.
Adds volume confirmation (volume > 1.5x 20-bar average) to avoid false breakouts.
Uses 0.20 position size to limit drawdown. Designed for low trade frequency by requiring
multi-timeframe alignment and volume confirmation. Works in bull markets via trend following
and in bear markets via short signals during downtrends.
"""

name = "1h_4h1d_EMA_Crossover_Signal"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 34 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h EMA34 Trend Filter ---
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # --- 1d EMA50 Trend Filter ---
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- 1h EMA12 and EMA26 for Entry ---
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # --- Volume Confirmation ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_12[i]) or np.isnan(ema_26[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        # Determine trend alignment
        bullish_trend = (close[i] > ema_34_4h_aligned[i]) and (close[i] > ema_50_1d_aligned[i])
        bearish_trend = (close[i] < ema_34_4h_aligned[i]) and (close[i] < ema_50_1d_aligned[i])
        
        if position == 0:
            # Long: bullish trend + EMA12 crosses above EMA26 + volume
            if bullish_trend and (ema_12[i] > ema_26[i]) and (ema_12[i-1] <= ema_26[i-1]) and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: bearish trend + EMA12 crosses below EMA26 + volume
            elif bearish_trend and (ema_12[i] < ema_26[i]) and (ema_12[i-1] >= ema_26[i-1]) and volume_spike:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: trend reversal or EMA cross in opposite direction
            if position == 1:
                # Exit long: bearish trend or EMA12 crosses below EMA26
                if bearish_trend or (ema_12[i] < ema_26[i] and ema_12[i-1] >= ema_26[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: bullish trend or EMA12 crosses above EMA26
                if bullish_trend or (ema_12[i] > ema_26[i] and ema_12[i-1] <= ema_26[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals