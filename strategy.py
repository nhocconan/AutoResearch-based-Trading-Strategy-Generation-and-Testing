#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion at 4h Bollinger Bands with 1d trend filter and volume confirmation
# Uses Bollinger Band mean reversion in ranging markets (common in 2025 BTC/ETH).
# 4h Bollinger Bands provide dynamic support/resistance. 1d EMA50 filters trend direction.
# Volume spike >1.5 confirms mean reversion bounces. Target: 20-40 trades/year.
# Works in bull via bounces off lower BB in uptrend, in bear via bounces off upper BB in downtrend.
name = "1h_BB_MeanReversion_4hBB_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Bollinger Bands (20, 2)
    close_4h = df_4h['close'].values
    ma_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_4h = ma_4h + 2 * std_4h
    lower_4h = ma_4h - 2 * std_4h
    
    # Align Bollinger Bands to 1h
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    ma_4h_aligned = align_htf_to_ltf(prices, df_4h, ma_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation - 24-period average volume (1 day at 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: bounce off lower 4h BB with uptrend and volume confirmation
            if (close[i] <= lower_4h_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.20
                position = 1
            # Short entry: bounce off upper 4h BB with downtrend and volume confirmation
            elif (close[i] >= upper_4h_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: return to 4h mean or trend fails
            if close[i] >= ma_4h_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: return to 4h mean or trend fails
            if close[i] <= ma_4h_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals