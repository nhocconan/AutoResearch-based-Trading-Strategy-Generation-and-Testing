#/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla levels, trend, ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels for previous day
    # R3, S3, R4, S4 (we use R3 and S3 for entries)
    pivot = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3
    range_val = high_1d[-1] - low_1d[-1]
    r3 = close_1d[-1] + range_val * 1.1 / 4
    s3 = close_1d[-1] - range_val * 1.1 / 4
    
    # Camarilla levels for current day (update as new 1d bar forms)
    pivot_series = (high_1d + low_1d + close_1d) / 3
    range_series = high_1d - low_1d
    r3_series = close_1d + range_series * 1.1 / 4
    s3_series = close_1d - range_series * 1.1 / 4
    
    # 1d EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # ATR for volatility filter (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Increased threshold for fewer trades
    
    # Align all 1d indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_series)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_series)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # EMA34 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_aligned[i]) or np.isnan(atr_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + 1d uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + 1d downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 or trend reverses
            if close[i] < s3_aligned[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R3 or trend reverses
            if close[i] > r3_aligned[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals