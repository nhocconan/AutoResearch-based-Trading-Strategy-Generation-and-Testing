#!/usr/bin/env python3
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
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Keltner Channel (20-period EMA +/- 2*ATR)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20 + (2 * atr)
    lower_keltner = ema20 - (2 * atr)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA20, ATR, volume MA, and EMA34
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(ema20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        upper_kelt = upper_keltner[i]
        lower_kelt = lower_keltner[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price closes above upper Keltner + volume spike + uptrend (price > EMA34)
            if close[i] > upper_kelt and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price closes below lower Keltner + volume spike + downtrend (price < EMA34)
            elif close[i] < lower_kelt and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below EMA20 or trend turns down
            if close[i] < ema20[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above EMA20 or trend turns up
            if close[i] > ema20[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Keltner_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0