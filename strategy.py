#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on 1d
    # Efficiency ratio: |close - close_10| / sum(|close - close_prev|) over 10 periods
    change = np.abs(np.diff(close, n=1))
    abs_change = np.abs(np.diff(close, n=1))
    er_num = np.abs(close[10:] - close[:-10])
    er_den = np.zeros_like(close)
    for i in range(10, len(close)):
        er_den[i] = np.sum(abs_change[i-9:i+1])
    er = np.zeros_like(close)
    er[10:] = er_num / (er_den + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 1d (already same timeframe, but for consistency)
    kama_aligned = kama  # no alignment needed for same timeframe
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi[14:]])  # align with original length
    
    # Calculate EMA(200) on 1w for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    # Align EMA(200) from 1w to 1d
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume filter: current volume > 20-period average (on 1d data)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend from 1w EMA200
        uptrend = close_1w[i] > ema_200_1w[i] if i < len(close_1w) else ema_200_1w_aligned[i] < close[i]
        downtrend = close_1w[i] < ema_200_1w[i] if i < len(close_1w) else ema_200_1w_aligned[i] > close[i]
        
        # Entry conditions
        long_signal = (close[i] > kama_aligned[i]) and (rsi[i] > 50) and volume_ok[i] and uptrend
        short_signal = (close[i] < kama_aligned[i]) and (rsi[i] < 50) and volume_ok[i] and downtrend
        
        # Exit conditions: reverse of entry
        exit_long = (close[i] < kama_aligned[i]) or (rsi[i] < 50)
        exit_short = (close[i] > kama_aligned[i]) or (rsi[i] > 50)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals