#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_12hTrend_VolumeSpike
Hypothesis: Keltner Channel breakouts with 12h EMA trend filter and volume spike (>1.5x 20-period average). 
Keltner Channels adapt to volatility, providing dynamic support/resistance. Breakouts above upper channel signal bullish momentum, 
breakouts below lower channel signal bearish momentum. 12h EMA ensures alignment with higher timeframe trend, reducing false signals. 
Volume spike confirms breakout strength. Designed for moderate trade frequency (15-25/year) with clear trend-following logic.
Works in bull/bear markets by requiring trend alignment and volatility-based channels.
"""

name = "4h_Keltner_Channel_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    daily_close = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Keltner Channel (20, 2.0)
    # Middle line: EMA(20) of close
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR(20)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Upper and lower bands
    kc_upper = ema_20 + 2.0 * atr
    kc_lower = ema_20 - 2.0 * atr
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend using aligned close
        daily_close_aligned = align_htf_to_ltf(prices, df_12h, daily_close)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = daily_close_aligned[i] > ema_50_12h_aligned[i]
        trend_down = daily_close_aligned[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: Close breaks above upper Keltner with uptrend and volume spike
            if close[i] > kc_upper[i] and trend_up and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Keltner with downtrend and volume spike
            elif close[i] < kc_lower[i] and trend_down and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close crosses below middle line or trend turns down
            if close[i] < ema_20[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close crosses above middle line or trend turns up
            if close[i] > ema_20[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals