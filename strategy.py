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
    
    # Get daily data for EMA trend filter and volume SMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily volume SMA50 for volume filter
    vol_sma50_1d = pd.Series(df_1d['volume']).rolling(window=50, min_periods=50).mean().values
    vol_sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma50_1d)
    
    # Calculate ATR (14-period) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Bollinger Bands (20-period SMA, 2 std dev)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for SMA20, std20, ATR, and daily indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma50_1d_aligned[i]) or \
           np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        daily_trend = ema50_1d_aligned[i]
        daily_vol_sma = vol_sma50_1d_aligned[i]
        upper_bb_val = upper_bb[i]
        lower_bb_val = lower_bb[i]
        
        if position == 0:
            # Long: price touches lower Bollinger Band + volume above daily average + uptrend (close > daily EMA50)
            if low[i] <= lower_bb_val and volume[i] > daily_vol_sma and close[i] > daily_trend:
                signals[i] = size
                position = 1
            # Short: price touches upper Bollinger Band + volume above daily average + downtrend (close < daily EMA50)
            elif high[i] >= upper_bb_val and volume[i] > daily_vol_sma and close[i] < daily_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above SMA20 or trend turns down
            if close[i] > sma20[i] or close[i] < daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses below SMA20 or trend turns up
            if close[i] < sma20[i] or close[i] > daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Bollinger_Touch_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0