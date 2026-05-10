#!/usr/bin/env python3
"""
6h_RSI2_Trend_With_Volume
Hypothesis: 2-period RSI extreme pullbacks in the direction of 1d trend, with volume confirmation.
Works in bull/bear markets by only taking long in uptrend, short in downtrend.
Low trade frequency (<30/year) due to strict RSI(2) thresholds + trend filter + volume.
"""

name = "6h_RSI2_Trend_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 6h data for RSI(2) and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(2) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI(2) period + EMA34
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: price vs 1d EMA34
        bullish_trend = close[i] > ema_34_1d_aligned[i]
        bearish_trend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: bullish trend AND RSI(2) oversold (<10) with volume
            if bullish_trend and rsi[i] < 10 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish trend AND RSI(2) overbought (>90) with volume
            elif bearish_trend and rsi[i] > 90 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI(2) overbought (>80) OR trend turns bearish
            if rsi[i] > 80 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI(2) oversold (<20) OR trend turns bullish
            if rsi[i] < 20 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals