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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR for Keltner channel (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period EMA for Keltner middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel (20 EMA +/- 2*ATR)
    upper_keltner = ema20 + (2 * atr)
    lower_keltner = ema20 - (2 * atr)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA20, ATR(14), volume MA, and EMA50
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(ema20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        upper_kelt = upper_keltner[i]
        lower_kelt = lower_keltner[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price closes above upper Keltner + volume spike + uptrend (price > EMA50)
            if close[i] > upper_kelt and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price closes below lower Keltner + volume spike + downtrend (price < EMA50)
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

name = "4h_Keltner_Breakout_Volume_Trend_v2"
timeframe = "4h"
leverage = 1.0