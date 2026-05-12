#!/usr/bin/env python3
name = "1d_WeeklyTrend_DailyPullback_Entry_v4"
timeframe = "1d"
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
    
    # ===== Weekly Trend Filter (HTF) =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ===== Daily Pullback Entry (LTF) =====
    ema20_d = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    rsi14 = compute_rsi(close, 14)
    atr14 = compute_atr(high, low, close, 14)
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema20_d[i]) or np.isnan(rsi14[i]) or np.isnan(atr14[i])):
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
            # Long: Weekly uptrend + price pulls back to EMA20 + RSI < 40
            if (close[i] > ema50_1w_aligned[i] and
                low[i] <= ema20_d[i] and
                rsi14[i] < 40):
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + price pulls back to EMA20 + RSI > 60
            elif (close[i] < ema50_1w_aligned[i] and
                  high[i] >= ema20_d[i] and
                  rsi14[i] > 60):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Weekly trend breaks or RSI > 70
            if (close[i] < ema50_1w_aligned[i] or
                rsi14[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Weekly trend breaks or RSI < 30
            if (close[i] > ema50_1w_aligned[i] or
                rsi14[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def compute_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    atr[period] = np.mean(tr[:period])
    
    for i in range(period + 1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    atr[:period] = np.nan
    return atr