#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 200-day EMA trend filter with RSI(14) mean reversion
# and volume confirmation. Works in bull/bear by only taking mean-reversion trades
# in the direction of the long-term trend. Weekly trend filter avoids counter-trend
# whipsaws. Target 15-25 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 200 EMA and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 200-day EMA for trend filter
    ema_200_1d = np.full(len(df_1d), np.nan)
    alpha = 2 / (200 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_200_1d[i] = close_1d[i]
        elif np.isnan(ema_200_1d[i-1]):
            ema_200_1d[i] = np.mean(close_1d[:i+1])
        else:
            ema_200_1d[i] = close_1d[i] * alpha + ema_200_1d[i-1] * (1 - alpha)
    
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate daily ATR(14) for stop loss and volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Get weekly data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-week EMA for trend filter
    ema_50_1w = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (50 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_50_1w[i] = close_1w[i]
        elif np.isnan(ema_50_1w[i-1]):
            ema_50_1w[i] = np.mean(close_1w[:i+1])
        else:
            ema_50_1w[i] = close_1w[i] * alpha_w + ema_50_1w[i-1] * (1 - alpha_w)
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_14 = np.full(n, np.nan)
    rsi_14[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    # Volume filter: today's volume > 1.5x average volume
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(200, 50, 14, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(rsi_14[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filters: price above 200 EMA and weekly EMA above its previous value
        long_trend = (price > ema_200_1d_aligned[i]) and (ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1])
        short_trend = (price < ema_200_1d_aligned[i]) and (ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1])
        
        if position == 0:
            # Long: RSI < 30 (oversold) + long trend + volume
            if (rsi_14[i] < 30 and 
                long_trend and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + short trend + volume
            elif (rsi_14[i] > 70 and 
                  short_trend and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 or trend turns down
            if (rsi_14[i] > 70 or 
                ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 30 or trend turns up
            if (rsi_14[i] < 30 or 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA200_RSI14_Volume_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0