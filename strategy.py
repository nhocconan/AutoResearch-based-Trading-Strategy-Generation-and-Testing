#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Connors RSI with 1d trend filter and volume confirmation
# Connors RSI (CRSI) combines short-term RSI, streak RSI, and percentile rank to identify
# extreme overbought/oversold conditions. We use CRSI < 15 for longs and > 85 for shorts.
# Trades are filtered by 1d ADX > 25 (strong trend) and 1d volume > 1.5x 20-period average.
# This strategy aims to capture mean-reversion bounces within strong trends, working in
# both bull and bear markets by only trading in the direction of the trend.
# Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.

name = "6h_CRSI_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    tr = np.zeros(len(high_1d))
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
    
    # Wilder smoothing function
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    adx_strong = adx > 25
    adx_strong_6h = align_htf_to_ltf(prices, df_1d, adx_strong)
    
    # Calculate 1d volume average (20-period) and spike detection
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_spike_1d = df_1d['volume'].values > (vol_ma_1d.values * 1.5)
    vol_spike_6h = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Calculate Connors RSI (CRSI) on 6h timeframe
    # CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    # RSI(3)
    def rsi(arr, period):
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    rsi_3 = rsi(close, 3)
    
    # Streak RSI(2)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    rsi_streak = rsi(streak_abs, 2)
    
    # PercentRank(100) - percentage of values in last 100 periods below current value
    def percentile_rank(arr, window):
        rank = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window:
                rank[i] = np.nan
            else:
                window_data = arr[i-window:i]
                rank[i] = np.sum(window_data < arr[i]) / window * 100
        return rank
    
    percent_rank = percentile_rank(close, 100)
    
    # CRSI
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient data for CRSI calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(crsi[i]) or np.isnan(adx_strong_6h[i]) or np.isnan(vol_spike_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: CRSI < 15 (oversold), strong trend, volume spike
            if crsi[i] < 15 and adx_strong_6h[i] and vol_spike_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: CRSI > 85 (overbought), strong trend, volume spike
            elif crsi[i] > 85 and adx_strong_6h[i] and vol_spike_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CRSI > 70 (overbought) or trend weakens
            if crsi[i] > 70 or not adx_strong_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CRSI < 30 (oversold) or trend weakens
            if crsi[i] < 30 or not adx_strong_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals