#!/usr/bin/env python3
"""
4h CRSI with 1d Trend Filter and Volume Spike
Long: CRSI < 15 + price > 1d EMA(200) + volume > 1.5x 4h volume SMA(20)
Short: CRSI > 85 + price < 1d EMA(200) + volume > 1.5x 4h volume SMA(20)
Exit: CRSI crosses above 50 (long) or below 50 (short)
Uses CRSI (Connors RSI) for mean reversion, 1d EMA for trend filter, volume for confirmation
Target: 15-25 trades/year per symbol (60-100 total over 4 years)
"""

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
    
    # Calculate RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = np.mean(gain[:14]) if len(gain) >= 14 else 0
    avg_loss[0] = np.mean(loss[:14]) if len(loss) >= 14 else 0
    
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi3 = 100 - (100 / (1 + rs))
    
    # Calculate RSI(2) for streak
    up_streak = np.zeros(n)
    down_streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            up_streak[i] = up_streak[i-1] + 1
            down_streak[i] = 0
        elif close[i] < close[i-1]:
            down_streak[i] = down_streak[i-1] + 1
            up_streak[i] = 0
        else:
            up_streak[i] = 0
            down_streak[i] = 0
    
    # RSI of up streak (2-period)
    up_streak_rsi = np.zeros(n)
    down_streak_rsi = np.zeros(n)
    for i in range(n):
        if up_streak[i] >= 2:
            # Calculate RSI for up streak
            streak_gains = np.minimum(up_streak[i], 2)
            streak_losses = 0
            if streak_losses == 0:
                up_streak_rsi[i] = 100
            else:
                rs_streak = streak_gains / streak_losses
                up_streak_rsi[i] = 100 - (100 / (1 + rs_streak))
        elif down_streak[i] >= 2:
            streak_gains = 0
            streak_losses = np.minimum(down_streak[i], 2)
            if streak_gains == 0:
                down_streak_rsi[i] = 0
            else:
                rs_streak = streak_gains / streak_losses
                down_streak_rsi[i] = 100 - (100 / (1 + rs_streak))
        else:
            up_streak_rsi[i] = 50
            down_streak_rsi[i] = 50
    
    # RSI of percentile rank (100-period lookback)
    def percentile_rank(arr, window):
        rank = np.full_like(arr, np.nan)
        for i in range(window, len(arr)):
            window_data = arr[i-window:i+1]
            rank[i] = np.sum(window_data <= arr[i]) / window * 100
        return rank
    
    rsi_streak = np.where(up_streak > down_streak, up_streak_rsi, down_streak_rsi)
    percent_rank = percentile_rank(rsi3, 100)
    
    # CRSI = (RSI(3) + RSI(Streak) + PercentRank(100)) / 3
    crsi = (rsi3 + rsi_streak + percent_rank) / 3
    
    # Get 1d data for EMA(200) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(200)
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4h volume SMA(20)
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(100, 200)  # need CRSI and EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(crsi[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_sma_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        ema_200_val = ema_200_1d_aligned[i]
        crsi_val = crsi[i]
        
        if position == 0:
            # Long: CRSI < 15 + price > 1d EMA200 + volume > 1.5x SMA
            if crsi_val < 15 and price > ema_200_val and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: CRSI > 85 + price < 1d EMA200 + volume > 1.5x SMA
            elif crsi_val > 85 and price < ema_200_val and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CRSI crosses above 50
            if crsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CRSI crosses below 50
            if crsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CRSI_TrendFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0