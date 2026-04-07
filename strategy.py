# 6H Fibonacci Retracement with 1D Trend Filter and Volume Confirmation
# Enters on pullbacks to 61.8% Fibonacci level during strong 1D trends with volume confirmation
# Works in bull markets (buy dips) and bear markets (sell rallies)
# Uses actual swing points for dynamic Fibonacci levels (not fixed lookback)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_fib_retracement_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D trend filter (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Dynamic Fibonacci levels based on recent swing points ===
    lookback = 50  # Look for swings in last 50 periods
    fib_levels = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Find recent swing high and low
        window_high = np.max(high[i-lookback:i])
        window_low = np.min(low[i-lookback:i])
        swing_range = window_high - window_low
        
        if swing_range > 0:
            # 61.8% retracement level
            fib_618 = window_low + 0.618 * swing_range
            fib_levels[i] = fib_618
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(fib_levels[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 61.8% level or trend reverses
            if close[i] < fib_levels[i] or ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 61.8% level or trend reverses
            if close[i] > fib_levels[i] or ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if vol_ratio[i] < 1.3:
                signals[i] = 0.0
                continue
            
            # Entry: Pullback to 61.8% Fib level in direction of 1D trend
            if close[i] <= fib_levels[i] * 1.005 and close[i] >= fib_levels[i] * 0.995:
                if ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                    # Uptrend: buy the dip
                    position = 1
                    signals[i] = 0.25
                elif ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                    # Downtrend: sell the rally
                    position = -1
                    signals[i] = -0.25
    
    return signals