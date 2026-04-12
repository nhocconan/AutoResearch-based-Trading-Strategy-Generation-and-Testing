#!/usr/bin/env python3
"""
6h_1d_weekly_bias_daily_trend
Hypothesis: In 6h timeframe, use 1d trend direction (EMA50) as bias, and weekly structure (Donchian high/low) for breakout confirmation.
Long when: price > EMA50(1d) AND breaks above weekly Donchian high (20-period) with volume confirmation.
Short when: price < EMA50(1d) AND breaks below weekly Donchian low with volume confirmation.
Exit when price returns to EMA50(1d) or volatility expands (weekly ATR expansion).
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drift.
Works in bull markets (trend-following breaks) and bear markets (mean-reversion at extremes via EMA filter).
"""

name = "6h_1d_weekly_bias_daily_trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA bias and weekly structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # EMA50 on 1d for trend bias
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly Donchian channels (20-period ~ 20 days)
    donch_len = 20
    donch_high = pd.Series(high_1d).rolling(window=donch_len, min_periods=donch_len).max().values
    donch_low = pd.Series(low_1d).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Weekly ATR for volatility filter (14-period)
    atr_len = 14
    tr1 = pd.Series(high_1d).rolling(2).max().values - pd.Series(low_1d).rolling(2).min().values
    tr2 = abs(pd.Series(high_1d).rolling(2).shift(1).values - pd.Series(close_1d).rolling(2).shift(1).values)
    tr3 = abs(pd.Series(low_1d).rolling(2).shift(1).values - pd.Series(close_1d).rolling(2).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_len, min_periods=atr_len).mean().values
    
    # Align 1d indicators to 6h
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation on 6f: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price > EMA50(1d) AND breaks above weekly Donchian high with volume
        if (close[i] > ema50_aligned[i] and 
            close[i] > donch_high_aligned[i] and 
            vol_confirm[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price < EMA50(1d) AND breaks below weekly Donchian low with volume
        elif (close[i] < ema50_aligned[i] and 
              close[i] < donch_low_aligned[i] and 
              vol_confirm[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and close[i] <= ema50_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= ema50_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals