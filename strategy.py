#!/usr/bin/env python3
"""
1h_RSI_4H_Trend_1D_Volume_Confirm
Hypothesis: On 1h timeframe, use RSI(14) for entry timing (RSI<30 long, RSI>70 short) filtered by 4h EMA50 trend and 1d volume surge. This captures mean reversion within the trend, reducing whipsaw. 4h/1d filters ensure we trade with higher timeframe momentum, limiting trades to 15-35/year to avoid fee drag. Works in bull/bear via trend filter.
"""

name = "1h_RSI_4H_Trend_1D_Volume_Confirm"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 1h data for RSI and price
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 4h EMA50
        uptrend_4h = close[i] > ema50_4h_aligned[i]
        downtrend_4h = close[i] < ema50_4h_aligned[i]
        
        # Volume filter: current 1h volume > 1.5x 1d 20-period MA
        volume_filter = volume[i] > vol_ma20_1d_aligned[i] * 1.5
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: RSI oversold in uptrend with volume
            if rsi[i] < 30 and uptrend_4h and volume_filter and in_session:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought in downtrend with volume
            elif rsi[i] > 70 and downtrend_4h and volume_filter and in_session:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or trend fails
            if rsi[i] > 70 or not uptrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI oversold or trend fails
            if rsi[i] < 30 or not downtrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals