#!/usr/bin/env python3
# 4h_KC_Breakout_Volume_KC_Width
# Hypothesis: Bollinger Bandwidth (BBW) identifies low-volatility squeezes that precede breakouts.
# When price breaks above the upper Keltner Channel (KC) with volume confirmation, it signals a bullish breakout.
# Conversely, breaking below the lower KC with volume confirms a bearish breakout.
# KC is used because it adapts better to volatility than BB, reducing false breakouts in choppy markets.
# Volume filter ensures breakouts are supported by participation, reducing false signals.
# Designed for low trade frequency (20-40/year) to minimize fee drag.

name = "4h_KC_Breakout_Volume_KC_Width"
timeframe = "4h"
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
    
    # Bollinger Bands for bandwidth calculation (20, 2)
    bb_period = 20
    bb_mult = 2
    close_series = pd.Series(close)
    bb_ma = close_series.ewm(span=bb_period, adjust=False, min_periods=bb_period).mean()
    bb_std = close_series.ewm(span=bb_period, adjust=False, min_periods=bb_period).std()
    bb_upper = bb_ma + bb_mult * bb_std
    bb_lower = bb_ma - bb_mult * bb_std
    bb_width = (bb_upper - bb_lower) / bb_ma  # Normalized bandwidth
    
    # Keltner Channel (20, ATR multiplier 1.5)
    kc_period = 20
    kc_mult = 1.5
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # ATR
    atr = pd.Series(tr).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean()
    
    # KC middle line (EMA of close)
    kc_middle = close_series.ewm(span=kc_period, adjust=False, min_periods=kc_period).mean()
    kc_upper = kc_middle + kc_mult * atr
    kc_lower = kc_middle - kc_mult * atr
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    # Bollinger Bandwidth regime filter: low volatility = squeeze
    # We use 50-period percentile of BBW to define squeeze (below 20th percentile = squeeze)
    bbw_series = pd.Series(bb_width.values)
    bbw_rank = bbw_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    squeeze_condition = bbw_rank < 0.2  # Below 20th percentile = low volatility squeeze
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5  # Need enough history for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(squeeze_condition.iloc[i] if hasattr(squeeze_condition, 'iloc') else squeeze_condition[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Extract squeeze value safely
        sq_val = squeeze_condition.iloc[i] if hasattr(squeeze_condition, 'iloc') else squeeze_condition[i]
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long breakout: price closes above upper KC AND volatility squeeze
            if close[i] > kc_upper[i] and sq_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below lower KC AND volatility squeeze
            elif close[i] < kc_lower[i] and sq_val and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below middle KC OR volatility expansion (end of squeeze)
            if close[i] < kc_middle[i] or not sq_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above middle KC OR volatility expansion
            if close[i] > kc_middle[i] or not sq_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals