#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_WeeklyTrend_Filter_v1
Hypothesis: Use KAMA to capture trend direction on 1d timeframe, filtered by weekly trend (EMA34) and volume confirmation. KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing strong trends. Weekly trend filter ensures we only trade in the direction of the higher timeframe trend, improving win rate in both bull and bear markets. Target: 15-25 trades/year.
"""

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
    
    # KAMA calculation (adaptive moving average)
    # ER = Efficiency Ratio = abs(close - close[10]) / sum(abs(close - close[1:11]))
    # SSC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prevKAMA + SSC * (close - prevKAMA)
    lookback = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate change and volatility
    change = np.abs(np.diff(close, n=lookback))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1, volatility)
    er = np.concatenate([np.full(lookback, np.nan), change / volatility])
    
    # Calculate SSC
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[lookback] = close[lookback]  # Initialize
    
    for i in range(lookback + 1, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly trend filter: EMA34 on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Wait for KAMA initialization
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        weekly_trend = ema_34_1w_aligned[i]
        vol_conf = volume_confirmed[i]
        
        if position == 0:
            # Long: price above KAMA, above weekly trend, with volume confirmation
            if price > kama_val and price > weekly_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, below weekly trend, with volume confirmation
            elif price < kama_val and price < weekly_trend and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below KAMA or below weekly trend
            if price < kama_val or price < weekly_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above KAMA or above weekly trend
            if price > kama_val or price > weekly_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_WeeklyTrend_Filter_v1"
timeframe = "1d"
leverage = 1.0