# 1d_1w_kama_rsi_v1
# Hypothesis: Use weekly trend filter with daily KAMA direction and RSI for entries. In weekly uptrend: go long when KAMA turns up and RSI < 40; short when KAMA turns down and RSI > 60. In weekly downtrend: go short when KAMA turns down and RSI > 60; long when KAMA turns up and RSI < 40. Uses KAMA for trend-following entry with RSI for mean-reversion timing, targeting 8-20 trades/year (32-80 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA21 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Daily KAMA calculation
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility as sum of absolute changes over ER period
    er_period = 10
    change_abs = np.abs(np.diff(close, prepend=close[0]))
    volatility_sum = np.zeros_like(close)
    for i in range(er_period, len(close)):
        volatility_sum[i] = np.sum(change_abs[i-er_period+1:i+1])
    change_sum = np.abs(close - np.roll(close, er_period))
    change_sum[:er_period] = 0
    er = np.where(volatility_sum != 0, change_sum / volatility_sum, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly EMA21 to daily
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema21_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend
        weekly_uptrend = close[i] > ema21_1w_aligned[i]
        
        # KAMA direction (1=up, -1=down)
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        if position == 1:  # Long position
            # Exit: weekly trend breaks or KAMA turns down with RSI > 50
            if not weekly_uptrend or (kama_down and rsi[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly trend breaks or KAMA turns up with RSI < 50
            if weekly_uptrend or (kama_up and rsi[i] < 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: weekly uptrend, KAMA turns up, RSI < 40 (oversold)
            if weekly_uptrend and kama_up and rsi[i] < 40:
                position = 1
                signals[i] = 0.25
            # Short entry: weekly downtrend, KAMA turns down, RSI > 60 (overbought)
            elif not weekly_uptrend and kama_down and rsi[i] > 60:
                position = -1
                signals[i] = -0.25
    
    return signals