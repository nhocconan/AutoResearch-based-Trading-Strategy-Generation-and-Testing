#!/usr/bin/env python3
name = "4h_1d_RSI_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    
    # 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Get 4h data for entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h RSI(14)
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    
    # 4h Volume MA(20) for confirmation
    vol_ma20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    vol_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trend_up_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(rsi_4h_aligned[i]) or
            np.isnan(vol_ma20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 1d uptrend + 4h RSI < 30 (oversold) + volume confirmation
            if (trend_up_1d_aligned[i] and 
                rsi_4h_aligned[i] < 30 and 
                volume[i] > 1.5 * vol_ma20_4h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend + 4h RSI > 70 (overbought) + volume confirmation
            elif (not trend_up_1d_aligned[i] and 
                  rsi_4h_aligned[i] > 70 and 
                  volume[i] > 1.5 * vol_ma20_4h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 4h RSI > 70 (overbought) or trend change
            if (rsi_4h_aligned[i] > 70 or 
                not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 4h RSI < 30 (oversold) or trend change
            if (rsi_4h_aligned[i] < 30 or 
                trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals