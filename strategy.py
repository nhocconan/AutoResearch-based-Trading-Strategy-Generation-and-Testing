#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ConnersRsi_Contrarian_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d RSI(3) for CRSI component
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 3:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # RSI(3)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/3, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/3, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi3 = 100 - (100 / (1 + rs))
    
    # RSI streak (2)
    up_days = np.where(np.diff(close_1d, prepend=close_1d[0]) > 0, 1, 0)
    down_days = np.where(np.diff(close_1d, prepend=close_1d[0]) < 0, 1, 0)
    streak_up = np.where(up_days, np.add.accumulate(up_days) - np.add.accumulate(up_days * (1 - up_days)), 0)
    streak_down = np.where(down_days, np.add.accumulate(down_days) - np.add.accumulate(down_days * (1 - down_days)), 0)
    streak = np.where(streak_up > 0, streak_up, -streak_down)
    # RSI of streak
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    avg_sg = pd.Series(streak_gain).ewm(alpha=1/2, adjust=False).mean().values
    avg_sl = pd.Series(streak_loss).ewm(alpha=1/2, adjust=False).mean().values
    rs_s = avg_sg / (avg_sl + 1e-10)
    rsi_streak = 100 - (100 / (1 + rs_s))
    
    # Percent Rank (100)
    def percent_rank(arr, window):
        pr = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                pr[i] = np.nan
            else:
                window_data = arr[i-window+1:i+1]
                rank = np.sum(window_data <= arr[i]) / window * 100
                pr[i] = rank
        return pr
    pr100 = percent_rank(close_1d, 100)
    
    # CRSI = (RSI(3) + RSI_streak + PercentRank(100)) / 3
    crsi = (rsi3 + rsi_streak + pr100) / 3.0
    crsi = np.where(np.isnan(crsi), 50, crsi)  # neutral when undefined
    
    # Align CRSI to 4h
    crsi_aligned = align_htf_to_ltf(prices, df_1d, crsi)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(crsi_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CRSI < 15 (oversold) and price above 1d EMA34 and volume spike
            long_cond = (crsi_aligned[i] < 15 and 
                        close[i] > ema34_1d_aligned[i] and
                        volume_spike[i])
            
            # Short: CRSI > 85 (overbought) and price below 1d EMA34 and volume spike
            short_cond = (crsi_aligned[i] > 85 and 
                         close[i] < ema34_1d_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CRSI > 70 (overbought) or price crosses below 1d EMA34
            if crsi_aligned[i] > 70 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CRSI < 30 (oversold) or price crosses above 1d EMA34
            if crsi_aligned[i] < 30 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals