# 1d_KAMA_Trend_Filter_With_Volume_Spike
# Hypothesis: KAMA trend direction on 1d with volume spike confirmation and 1w trend filter.
# Buy when KAMA turns up with volume spike and weekly uptrend; short when KAMA turns down with volume spike and weekly downtrend.
# Uses only 3 conditions (KAMA turn, volume spike, weekly trend) to keep trades ~15-25/year.
# Designed for 1d timeframe to avoid overtrading and work in both bull and bear markets via trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Fix: volatility should be rolling sum of changes
    volatility = pd.Series(change).rolling(window=er_length, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA
    df_1d = get_htf_data(prices, '1d')
    kama = calculate_kama(df_1d['close'].values)
    kama_1d = align_htf_to_ltf(prices, df_1d, kama)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Simple trend: price above/below 50-period EMA on weekly
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Warmup for weekly EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_1d[i]
        ema50w = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Detect KAMA turning points (using 2-bar lookback for confirmation)
        if i >= 2:
            kama_prev = kama_1d[i-1]
            kama_prev2 = kama_1d[i-2]
            kama_rising = kama_val > kama_prev and kama_prev > kama_prev2
            kama_falling = kama_val < kama_prev and kama_prev < kama_prev2
        else:
            kama_rising = False
            kama_falling = False
        
        if position == 0:
            # Long: KAMA turning up with volume spike and weekly uptrend
            if kama_rising and vol_spike and price > ema50w:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down with volume spike and weekly downtrend
            elif kama_falling and vol_spike and price < ema50w:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: KAMA turns down OR weekly trend turns down
            if kama_falling:
                signals[i] = 0.0
                position = 0
            elif price < ema50w:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: KAMA turns up OR weekly trend turns up
            if kama_rising:
                signals[i] = 0.0
                position = 0
            elif price > ema50w:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Filter_With_Volume_Spike"
timeframe = "1d"
leverage = 1.0