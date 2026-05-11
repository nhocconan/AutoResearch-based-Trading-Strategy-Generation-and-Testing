#!/usr/bin/env python3
name = "12h_RSI_Correction_with_Volume_and_Trend_Filter"
timeframe = "12h"
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
    
    # Get 1d data for trend filter and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(delta)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:13] = np.nan  # Not enough data for first 13 periods
    
    # Calculate 1d EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    
    # Get 12h data for entry signals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h RSI(14) for mean reversion signals
    delta_12h = np.diff(close_12h, prepend=close_12h[0])
    gain_12h = np.where(delta_12h > 0, delta_12h, 0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0)
    
    avg_gain_12h = np.zeros_like(gain_12h)
    avg_loss_12h = np.zeros_like(loss_12h)
    avg_gain_12h[13] = np.mean(gain_12h[1:14])
    avg_loss_12h[13] = np.mean(loss_12h[1:14])
    
    for i in range(14, len(delta_12h)):
        avg_gain_12h[i] = (avg_gain_12h[i-1] * 13 + gain_12h[i]) / 14
        avg_loss_12h[i] = (avg_loss_12h[i-1] * 13 + loss_12h[i]) / 14
    
    rs_12h = np.divide(avg_gain_12h, avg_loss_12h, out=np.full_like(avg_gain_12h, np.nan), where=avg_loss_12h!=0)
    rsi_12h = 100 - (100 / (1 + rs_12h))
    rsi_12h[:13] = np.nan
    
    # Calculate 12h ATR(14) for volatility filter
    tr_12h = np.zeros(len(df_12h))
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(df_12h)):
        tr = high_12h[i] - low_12h[i]
        tr2 = abs(high_12h[i] - close_12h[i-1])
        tr3 = abs(low_12h[i] - close_12h[i-1])
        tr_12h[i] = max(tr, tr2, tr3)
    
    atr14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    atr14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr14_12h)
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            if i > 0:
                vol_ma20[i] = np.mean(volume[:i+1])
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(rsi_12h_aligned[i]) or
            np.isnan(atr14_12h_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 1d uptrend + 12h RSI oversold + volume confirmation
            if (trend_up_1d_aligned[i] and 
                rsi_12h_aligned[i] < 30 and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend + 12h RSI overbought + volume confirmation
            elif (not trend_up_1d_aligned[i] and 
                  rsi_12h_aligned[i] > 70 and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or trend change
            if (rsi_12h_aligned[i] > 70 or 
                not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or trend change
            if (rsi_12h_aligned[i] < 30 or 
                trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals