#!/usr/bin/env python3
name = "6h_RSI_Bollinger_Band_Squeeze"
timeframe = "6h"
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
    
    # Get 1w data for RSI(14) - weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    rsi_1w = np.where(rsi_1w < 0, 0, rsi_1w)
    rsi_1w = np.where(rsi_1w > 100, 100, rsi_1w)
    
    # Weekly trend: RSI > 50 = uptrend, RSI < 50 = downtrend
    trend_up_1w = rsi_1w > 50
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # Bollinger Bands on 6h (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = (bb_upper - bb_lower) / sma20
    
    # Bollinger Squeeze: width < 20th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze = bb_width < bb_width_percentile
    
    # RSI(14) on 6h for mean reversion
    delta_6h = np.diff(close, prepend=close[0])
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    
    avg_gain_6h = np.zeros_like(gain_6h)
    avg_loss_6h = np.zeros_like(loss_6h)
    avg_gain_6h[13] = np.mean(gain_6h[1:14])
    avg_loss_6h[13] = np.mean(loss_6h[1:14])
    
    for i in range(14, len(gain_6h)):
        avg_gain_6h[i] = (avg_gain_6h[i-1] * 13 + gain_6h[i]) / 14
        avg_loss_6h[i] = (avg_loss_6h[i-1] * 13 + loss_6h[i]) / 14
    
    rs_6h = np.where(avg_loss_6h != 0, avg_gain_6h / avg_loss_6h, 100)
    rsi_6h = np.where(avg_loss_6h == 0, 100, 100 - (100 / (1 + rs_6h)))
    rsi_6h = np.where(rsi_6h < 0, 0, rsi_6h)
    rsi_6h = np.where(rsi_6h > 100, 100, rsi_6h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(rsi_6h[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: squeeze breakout up + weekly uptrend + RSI oversold + volume
            if (close[i] > bb_upper[i-1] and trend_up_1w_aligned[i] and 
                rsi_6h[i] < 30 and volume_filter[i] and squeeze[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down + weekly downtrend + RSI overbought + volume
            elif (close[i] < bb_lower[i-1] and not trend_up_1w_aligned[i] and 
                  rsi_6h[i] > 70 and volume_filter[i] and squeeze[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or squeeze break down
            if rsi_6h[i] > 70 or close[i] < bb_lower[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or squeeze break up
            if rsi_6h[i] < 30 or close[i] > bb_upper[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals