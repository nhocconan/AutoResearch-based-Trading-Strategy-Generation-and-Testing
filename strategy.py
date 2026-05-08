#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h CRSI (Connors RSI) + 1d KAMA trend filter + volume spike confirmation.
# Long when CRSI < 15 AND price > KAMA(1d) AND volume > 1.5x 20-period average.
# Short when CRSI > 85 AND price < KAMA(1d) AND volume > 1.5x 20-period average.
# Exit when CRSI crosses back above 50 (long) or below 50 (short).
# CRSI captures short-term mean reversion with trend bias. KAMA adapts to market noise.
# Volume spike confirms institutional participation. Target: 60-100 total trades over 4 years (15-25/year).

name = "4h_CRSI_KAMA_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # CRSI calculation: (RSI(3) + RSI_STREAK(2) + PERCENT_RANK(100)) / 3
    # RSI(3)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    avg_loss = loss.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    rs = avg_gain / avg_loss
    rsi3 = 100 - (100 / (1 + rs))
    
    # RSI Streak (2-period)
    up_days = (close > np.roll(close, 1)).astype(int)
    down_days = (close < np.roll(close, 1)).astype(int)
    streak = np.where(up_days, 1, np.where(down_days, -1, 0))
    streak_sum = np.where(streak > 0, np.maximum.accumulate(streak * up_days), 
                          np.where(streak < 0, np.minimum.accumulate(streak * down_days), 0))
    rsi_streak = 100 - (100 / (1 + np.exp(streak_sum)))
    
    # Percent Rank (100-period)
    close_series = pd.Series(close)
    percent_rank = close_series.rolling(window=100, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # CRSI
    crsi = (rsi3.values + rsi_streak + percent_rank) / 3
    
    # 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA calculation (simplified using ER and smoothing constants)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, k=1)), axis=1)  # 1-period volatility
    er = np.where(volatility > 0, change / volatility, 0)
    # Pad ER to match length
    er = np.concatenate([np.full(9, np.nan), er]) if len(er) == len(close_1d) - 9 else er
    er = np.where(np.isnan(er), 0, er)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align 1d KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for percent rank
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(crsi[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: CRSI < 15, price > KAMA, volume spike
            long_cond = (crsi[i] < 15) and (close[i] > kama_aligned[i]) and volume_filter[i]
            # Short conditions: CRSI > 85, price < KAMA, volume spike
            short_cond = (crsi[i] > 85) and (close[i] < kama_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CRSI crosses above 50
            if crsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CRSI crosses below 50
            if crsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals