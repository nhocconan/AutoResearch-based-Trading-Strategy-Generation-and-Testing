#!/usr/bin/env python3
name = "1h_RSI_MACD_Trend_Follow_v1"
timeframe = "1h"
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
    
    # Get 4h trend (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_4h = close_4h > ema50_4h
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h)
    
    # Get daily trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up_1d = close_1d > ema200_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # RSI (14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # MACD (12,26,9) on 1h
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd = ema12 - ema26
    signal_line = pd.Series(macd).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd - signal_line
    
    # Volume filter
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 26, 9, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is invalid
        if (np.isnan(trend_up_4h_aligned[i]) or 
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(macd_hist[i]) or
            np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 40 (not oversold), MACD histogram turning up, aligned uptrend
            if (rsi[i] < 40 and 
                macd_hist[i] > macd_hist[i-1] and 
                trend_up_4h_aligned[i] and 
                trend_up_1d_aligned[i] and 
                volume[i] > 1.2 * vol_ma20[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 60 (not overbought), MACD histogram turning down, aligned downtrend
            elif (rsi[i] > 60 and 
                  macd_hist[i] < macd_hist[i-1] and 
                  not trend_up_4h_aligned[i] and 
                  not trend_up_1d_aligned[i] and 
                  volume[i] > 1.2 * vol_ma20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI > 70 or MACD histogram turns down or trend breaks
            if (rsi[i] > 70 or 
                macd_hist[i] < macd_hist[i-1] or 
                not trend_up_4h_aligned[i] or 
                not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI < 30 or MACD histogram turns up or trend breaks
            if (rsi[i] < 30 or 
                macd_hist[i] > macd_hist[i-1] or 
                trend_up_4h_aligned[i] or 
                trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals