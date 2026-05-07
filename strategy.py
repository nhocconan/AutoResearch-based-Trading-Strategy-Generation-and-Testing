#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_30mBias"
timeframe = "4h"
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
    open_price = prices['open'].values
    
    # Get 1d data for trend filter (EMA34) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d candle
    # Camarilla: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # We'll use R3 and S3 levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.roll(close_1d, 1)  # Previous day close
    high_1d_shifted = np.roll(high_1d, 1)   # Previous day high
    low_1d_shifted = np.roll(low_1d, 1)     # Previous day low
    
    # Calculate Camarilla levels for previous day
    camarilla_width = (high_1d_shifted - low_1d_shifted) * 1.1 / 4
    r3 = close_1d_shifted + camarilla_width  # R3 level
    s3 = close_1d_shifted - camarilla_width  # S3 level
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume confirmation (current volume vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    # Get 30m data for bias filter
    df_30m = get_htf_data(prices, '30m')
    if len(df_30m) < 14:
        return np.zeros(n)
    
    # Calculate 30m RSI(14) for bias filter
    close_30m = df_30m['close'].values
    delta = np.diff(close_30m, prepend=close_30m[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_30m = 100 - (100 / (1 + rs))
    rsi_30m_aligned = align_htf_to_ltf(prices, df_30m, rsi_30m)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(volume_ratio[i]) or 
            np.isnan(rsi_30m_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 level, uptrend (price > EMA34), volume confirmation, bullish bias (RSI > 50)
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_ratio[i] > 1.5 and 
                rsi_30m_aligned[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 level, downtrend (price < EMA34), volume confirmation, bearish bias (RSI < 50)
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_ratio[i] > 1.5 and 
                  rsi_30m_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 level (reversal signal)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 level (reversal signal)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals