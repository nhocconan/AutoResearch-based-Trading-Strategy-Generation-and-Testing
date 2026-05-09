#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Channel_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_12h = close_12h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 12h ATR10 for Keltner channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    
    tr1 = np.maximum(high_12h - low_12h, np.abs(high_12h - np.roll(close_12h_arr, 1)))
    tr2 = np.maximum(tr1, np.abs(low_12h - np.roll(close_12h_arr, 1)))
    tr2[0] = high_12h[0] - low_12h[0]  # First period
    atr_12h = pd.Series(tr2).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 4h Keltner channels using 12h EMA and ATR
    # We use the 12h EMA as the middle line, with 2*ATR bands
    keltner_middle = ema_12h_aligned
    keltner_upper = keltner_middle + 2 * atr_12h_aligned
    keltner_lower = keltner_middle - 2 * atr_12h_aligned
    
    # Volume spike detection (4h timeframe)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]  # Require strong volume spike
        
        if position == 0:
            # Long: Price breaks above upper Keltner band with uptrend and volume
            if close[i] > keltner_upper[i] and close[i] > keltner_middle[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Keltner band with downtrend and volume
            elif close[i] < keltner_lower[i] and close[i] < keltner_middle[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below middle line
            if close[i] < keltner_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above middle line
            if close[i] > keltner_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals