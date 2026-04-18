#!/usr/bin/env python3
"""
12h_Keltner_Channel_Touch_MeanReversion
Keltner Channel mean reversion with volume confirmation:
- Long when price touches lower Keltner Channel + volume spike
- Short when price touches upper Keltner Channel + volume spike
- Exit when price crosses middle EMA
- Uses 1d ATR for Keltner width and 1d EMA for trend filter
- Designed for 15-25 trades/year per symbol
Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(20)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = np.full(len(tr), np.nan)
    for i in range(20, len(tr)):
        atr_1d[i] = np.nanmean(tr[i-20+1:i+1])
    
    # Calculate 1d EMA(50)
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    atr_1d_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_1d_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h EMA(20) for middle line
    close_series = pd.Series(close)
    ema_20_12h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel parameters
    keltner_mult = 1.5
    
    # Calculate Keltner Channels
    upper_keltner = ema_20_12h + keltner_mult * atr_1d_12h
    lower_keltner = ema_20_12h - keltner_mult * atr_1d_12h
    
    # Volume spike detection (volume > 1.5 * 20-period average)
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema_50_1d_12h[i]) or np.isnan(ema_20_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches lower Keltner + volume spike + price below EMA50 (downtrend)
            if low[i] <= lower_keltner[i] and vol_spike[i] and close[i] < ema_50_1d_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper Keltner + volume spike + price above EMA50 (uptrend)
            elif high[i] >= upper_keltner[i] and vol_spike[i] and close[i] > ema_50_1d_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses above middle EMA(20)
            if close[i] > ema_20_12h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below middle EMA(20)
            if close[i] < ema_20_12h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Keltner_Channel_Touch_MeanReversion"
timeframe = "12h"
leverage = 1.0