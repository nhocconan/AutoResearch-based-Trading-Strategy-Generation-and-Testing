#!/usr/bin/env python3
"""
1h_RSI_Trend_Confluence
Hypothesis: RSI combined with 1d trend and volume confirmation on 1h timeframe.
Uses 1d trend (EMA200) for directional bias, RSI(14) for mean-reversion entries,
and volume spike for confirmation. Designed to work in both bull and bear markets
by filtering trades with the higher timeframe trend. Targets 15-35 trades/year
to minimize fee drag on 1h timeframe.
"""

name = "1h_RSI_Trend_Confluence"
timeframe = "1h"
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
    
    # Get 1d data for trend filter (called once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1d average volume for volume spike filter
    avg_vol_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # RSI(14) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20
    
    # Start after warmup for RSI
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) with uptrend (close > EMA200) and volume spike
            if (rsi[i] < 30 and 
                close[i] > ema200_1d_aligned[i] and 
                volume[i] > avg_vol_1d_aligned[i] * 1.5):
                signals[i] = position_size
                position = 1
            # Short: RSI > 70 (overbought) with downtrend (close < EMA200) and volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema200_1d_aligned[i] and 
                  volume[i] > avg_vol_1d_aligned[i] * 1.5):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60)
            if position == 1:
                if rsi[i] > 40:  # Exit long when RSI exits oversold
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if rsi[i] < 60:  # Exit short when RSI exits overbought
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals