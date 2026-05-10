#!/usr/bin/env python3
# 6h_Keltner_Breakout_Squeeze_Volume
# Hypothesis: Keltner Channel (KC) breakouts during low volatility squeezes with volume capture
# institutional participation. Squeeze identified by KC width percentile < 20% over 50 periods.
# Works in bull/bear as volatility expansion precedes directional moves regardless of trend.

name = "6h_Keltner_Breakout_Squeeze_Volume"
timeframe = "6h"
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
    
    # Keltner Channel (20, ATR multiplier 1.5)
    kc_period = 20
    kc_mult = 1.5
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR
    atr = pd.Series(tr).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # KC middle line (EMA of close)
    kc_middle = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    kc_upper = kc_middle + kc_mult * atr
    kc_lower = kc_middle - kc_mult * atr
    
    # KC Width for squeeze detection
    kc_width = (kc_upper - kc_lower) / kc_middle
    
    # Squeeze condition: KC width below 20th percentile over 50 periods
    kc_width_series = pd.Series(kc_width)
    kc_width_rank = kc_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).values
    squeeze_condition = kc_width_rank < 0.2
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, kc_period) + 5
    
    for i in range(start_idx, n):
        if np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(squeeze_condition[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long breakout: price closes above upper KC AND volatility squeeze
            if close[i] > kc_upper[i] and squeeze_condition[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below lower KC AND volatility squeeze
            elif close[i] < kc_lower[i] and squeeze_condition[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below middle KC OR volatility expansion (end of squeeze)
            if close[i] < kc_middle[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above middle KC OR volatility expansion
            if close[i] > kc_middle[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals