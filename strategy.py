#1d KAMA Trend with Weekly Filter v1
# Hypothesis: KAMA adapts to market noise, reducing whipsaw in ranging markets. Weekly trend filter ensures alignment with higher-timeframe momentum. Works in both bull and bear by following the weekly trend while filtering noise on daily.
# Expect 10-25 trades/year with low false signals due to dual timeframe confirmation.

#!/usr/bin/env python3
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
    
    # === DAILY KAMA CALCULATION ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.abs(np.diff(close, n=1))
    volatility = np.concatenate([np.array([np.nan]), volatility])
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = change / vol_sum
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === VOLUME CONFIRMATION (DAILY) ===
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === SIGNAL GENERATION ===
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(34, 20)  # weekly EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(weekly_trend[i]) or np.isnan(volume_ma20[i]):
            signals[i] = 0.0
            continue
        
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: price above KAMA AND above weekly EMA34 AND volume surge
            if close[i] > kama[i] and close[i] > weekly_trend[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA AND below weekly EMA34 AND volume surge
            elif close[i] < kama[i] and close[i] < weekly_trend[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA OR weekly trend turns down
            if close[i] < kama[i] or close[i] < weekly_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA OR weekly trend turns up
            if close[i] > kama[i] or close[i] > weekly_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_Weekly_Filter_v1"
timeframe = "1d"
leverage = 1.0