#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ConnorsRSI_DonchianBreakout_12hTrend"
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
    
    # 12h trend: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # RSI(3)
    def rsi(arr, period):
        delta = np.diff(arr)
        up = np.clip(delta, 0, None)
        down = np.clip(-delta, 0, None)
        ma_up = np.zeros_like(arr)
        ma_down = np.zeros_like(arr)
        ma_up[period] = np.mean(up[:period])
        ma_down[period] = np.mean(down[:period])
        for i in range(period + 1, len(arr)):
            ma_up[i] = (ma_up[i-1] * (period - 1) + up[i-1]) / period
            ma_down[i] = (ma_down[i-1] * (period - 1) + down[i-1]) / period
        rs = ma_up / (ma_down + 1e-10)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_3 = rsi(close, 3)
    
    # RSI(2) for streak
    rsi_2 = rsi(close, 2)
    # Streak: consecutive closes up/down
    streak_up = np.zeros(n)
    streak_down = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak_up[i] = streak_up[i-1] + 1
            streak_down[i] = 0
        elif close[i] < close[i-1]:
            streak_down[i] = streak_down[i-1] + 1
            streak_up[i] = 0
        else:
            streak_up[i] = 0
            streak_down[i] = 0
    # RSI of streak (2-period)
    def rsi_streak(arr, period):
        return rsi(arr, period)
    rsi_streak_up = rsi_streak(streak_up, 2)
    rsi_streak_down = rsi_streak(streak_down, 2)
    
    # Percent rank of RSI(3) over 100 periods
    def percent_rank(arr, lookback):
        pr = np.full_like(arr, np.nan)
        for i in range(lookback, len(arr)):
            window = arr[i-lookback:i]
            pr[i] = np.sum(window < arr[i]) / len(window) * 100
        return pr
    pr_rsi = percent_rank(rsi_3, 100)
    
    # Connors RSI (3-component)
    crsi = (rsi_3 + rsi_streak_up + rsi_streak_down + pr_rsi) / 4.0
    
    # Donchian(20)
    def donchian_channel(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_up, donch_dn = donchian_channel(high, low, 20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma20[:10] = np.nan
    vol_ma20[-10:] = np.nan
    volume_confirm = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(crsi[i]) or 
            np.isnan(donch_up[i]) or np.isnan(donch_dn[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CRSI < 15 (oversold) + price > Donchian upper + uptrend (price > 12h EMA50) + volume confirmation
            long_cond = (crsi[i] < 15) and \
                        (close[i] > donch_up[i]) and \
                        (close[i] > ema_50_12h_aligned[i]) and \
                        volume_confirm[i]
            # Short: CRSI > 85 (overbought) + price < Donchian lower + downtrend (price < 12h EMA50) + volume confirmation
            short_cond = (crsi[i] > 85) and \
                         (close[i] < donch_dn[i]) and \
                         (close[i] < ema_50_12h_aligned[i]) and \
                         volume_confirm[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CRSI > 70 (overbought) or price < Donchian lower
            if crsi[i] > 70 or close[i] < donch_dn[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CRSI < 30 (oversold) or price > Donchian upper
            if crsi[i] < 30 or close[i] > donch_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals