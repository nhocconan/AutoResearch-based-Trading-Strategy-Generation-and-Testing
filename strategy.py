#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour KAMA trend with daily volatility filter and volume confirmation
# We go long when KAMA turns upward (bullish trend) with low volatility regime and volume above average.
# We go short when KAMA turns downward (bearish trend) with low volatility regime and volume above average.
# Uses 4h timeframe targeting 20-50 trades/year. KAMA adapts to market noise, reducing false signals in chop.
# Daily volatility filter (low volatility regime) ensures we trade during calmer periods to avoid whipsaw.
# Volume confirmation ensures institutional participation in the move.

name = "4h_KAMA_Trend_DailyVol_Filter_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with 1d index
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma50 = pd.Series(atr14).rolling(window=50, min_periods=50).mean().values
    
    # Low volatility regime: current ATR < 50-period ATR average
    low_vol_regime = atr14 < atr_ma50
    
    # Align daily low volatility regime to 4h
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    # Calculate KAMA on 4h data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    abs_change = np.abs(np.diff(close, n=1))
    
    # Pad arrays for alignment
    change = np.concatenate([[np.nan]*10, change])
    abs_change = np.concatenate([[np.nan], abs_change])
    
    # Sum of absolute changes over 10 periods
    sum_abs_change = pd.Series(abs_change).rolling(window=10, min_periods=10).sum().values
    
    # Efficiency Ratio
    er = np.divide(change, sum_abs_change, out=np.zeros_like(change), where=sum_abs_change!=0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Fast=2, Slow=30
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: upward if current > previous, downward if current < previous
    kama_up = kama > np.roll(kama, 1)
    kama_down = kama < np.roll(kama, 1)
    kama_up[0] = False
    kama_down[0] = False
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(low_vol_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        low_vol = low_vol_aligned[i] > 0.5
        vol_conf = volume_conf[i]
        kama_up_val = kama_up[i]
        kama_down_val = kama_down[i]
        
        if position == 0:
            # Enter long: KAMA turning up + low volatility regime + volume confirmation
            if kama_up_val and low_vol and vol_conf:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA turning down + low volatility regime + volume confirmation
            elif kama_down_val and low_vol and vol_conf:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down OR volatility regime becomes high
            if kama_down_val or not low_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up OR volatility regime becomes high
            if kama_up_val or not low_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals