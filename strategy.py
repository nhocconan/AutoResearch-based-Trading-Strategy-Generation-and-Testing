#!/usr/bin/env python3
"""
4h_Keltner_Breakout_1dTrend_Volume
Hypothesis: Uses Keltner Channel (EMA20 + 2*ATR) breakouts with 1d EMA34 trend filter and volume confirmation to capture strong momentum moves. Keltner adapts to volatility better than fixed channels, reducing false signals in chop. Targets 20-30 trades/year on 4h to minimize fee drag while working in both bull and bear markets via trend filter.
"""

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(14) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    
    # Calculate EMA(20) for Keltner middle
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: upper = EMA20 + 2*ATR, lower = EMA20 - 2*ATR
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA20, ATR, EMA34, volume MA
    start_idx = max(20, 14, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(ema20[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        kc_up = kc_upper[i]
        kc_low = kc_lower[i]
        ema_mid = ema20[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: break above Keltner upper with uptrend and volume spike
            if close[i] > kc_up and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: break below Keltner lower with downtrend and volume spike
            elif close[i] < kc_low and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below Keltner middle or trend turns down
            if close[i] < ema_mid or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above Keltner middle or trend turns up
            if close[i] > ema_mid or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Keltner_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0