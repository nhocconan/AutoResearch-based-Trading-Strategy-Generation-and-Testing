#!/usr/bin/env python3
name = "1d_WeeklyTrend_DailyPullback_Entry_v3"
timeframe = "1d"
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
    
    # ===== Weekly Trend Filter (HTF) =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ===== Daily Pullback Entry =====
    ema20_d = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    rsi14_d = compute_rsi(close, 14)
    
    # ===== Daily Volume Spike Filter =====
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg_20)
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema20_d[i]) or np.isnan(rsi14_d[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Weekly uptrend + price pulls back to EMA20 + RSI < 40 + volume spike
            if (close[i] > ema50_1w_aligned[i] and
                low[i] <= ema20_d[i] and
                rsi14_d[i] < 40 and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + price bounces to EMA20 + RSI > 60 + volume spike
            elif (close[i] < ema50_1w_aligned[i] and
                  high[i] >= ema20_d[i] and
                  rsi14_d[i] > 60 and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Weekly trend breaks or RSI > 70
            if close[i] < ema50_1w_aligned[i] or rsi14_d[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Weekly trend breaks or RSI < 30
            if close[i] > ema50_1w_aligned[i] or rsi14_d[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def compute_rsi(prices, period=14):
    delta = np.diff(prices, prepend=prices[0])
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).ewm(alpha=1/period, adjust=False).mean().values
    roll_down = pd.Series(down).ewm(alpha=1/period, adjust=False).mean().values
    rs = roll_up / (roll_down + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi