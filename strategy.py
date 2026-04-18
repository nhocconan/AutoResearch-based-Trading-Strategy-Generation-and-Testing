#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI2_Streak_Trend_Filter"
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
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(2) on close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi2 = 100 - (100 / (1 + rs))
    rsi2_vals = rsi2.values
    
    # RSI(2) streak: consecutive days above/below 50
    # We'll calculate streak of closes above/below close 2 periods ago
    close_series = pd.Series(close)
    up_days = (close_series > close_series.shift(2)).astype(int)
    down_days = (close_series < close_series.shift(2)).astype(int)
    
    # Streak calculation
    up_streak = pd.Series(0, index=range(n))
    down_streak = pd.Series(0, index=range(n))
    
    for i in range(n):
        if i < 2:
            up_streak.iloc[i] = 0
            down_streak.iloc[i] = 0
        else:
            if up_days.iloc[i]:
                up_streak.iloc[i] = up_streak.iloc[i-1] + 1 if i > 0 else 1
                down_streak.iloc[i] = 0
            elif down_days.iloc[i]:
                down_streak.iloc[i] = down_streak.iloc[i-1] + 1 if i > 0 else 1
                up_streak.iloc[i] = 0
            else:
                up_streak.iloc[i] = 0
                down_streak.iloc[i] = 0
    
    up_streak_vals = up_streak.values
    down_streak_vals = down_streak.values
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 and RSI2
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi2_vals[i]) or
            np.isnan(up_streak_vals[i]) or np.isnan(down_streak_vals[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema50_val = ema50_1d_aligned[i]
        rsi2_val = rsi2_vals[i]
        up_streak_val = up_streak_vals[i]
        down_streak_val = down_streak_vals[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: RSI2 < 10, up streak >= 2, above daily EMA50, volume
            if rsi2_val < 10 and up_streak_val >= 2 and close_val > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: RSI2 > 90, down streak >= 2, below daily EMA50, volume
            elif rsi2_val > 90 and down_streak_val >= 2 and close_val < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI2 > 60 or price below EMA50
            if rsi2_val > 60 or close_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI2 < 40 or price above EMA50
            if rsi2_val < 40 or close_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals