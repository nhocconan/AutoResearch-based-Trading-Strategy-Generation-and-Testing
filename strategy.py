#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_7K_Pivot_Breakout_1wTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 7K pivot levels (1d)
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    
    # Calculate 7-period ATR for position sizing and volatility filter
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.absolute(np.roll(close, 1) - low)
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # First bar
    atr = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long entry: price breaks above R1 + uptrend (price > 1w EMA50) + volatility filter
            if high[i] > r1[i] and close[i] > ema_50_1w_aligned[i] and atr[i] > 0:
                # Size based on volatility (inverse ATR) with cap
                size = min(0.30, 0.05 * (np.median(atr) / atr[i])) if np.median(atr) > 0 else 0.25
                signals[i] = size
                position = 1
            # Short entry: price breaks below S1 + downtrend (price < 1w EMA50) + volatility filter
            elif low[i] < s1[i] and close[i] < ema_50_1w_aligned[i] and atr[i] > 0:
                size = min(0.30, 0.05 * (np.median(atr) / atr[i])) if np.median(atr) > 0 else 0.25
                signals[i] = -size
                position = -1
                
        elif position == 1:
            # Long position: exit on breakdown below S1 or trend reversal
            if low[i] < s1[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
                
        elif position == -1:
            # Short position: exit on breakout above R1 or trend reversal
            if high[i] > r1[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals