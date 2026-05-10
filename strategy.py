#!/usr/bin/env python3
# 4h_KC_Breakout_Squeeze_Volume_12hTrend
# Hypothesis: Keltner Channel breakouts during low volatility squeezes (BBW < 20th percentile) with volume confirmation,
# filtered by 12h EMA50 trend, produce high-quality breakouts with low frequency.
# Works in bull markets (breakouts with momentum) and bear markets (mean reversion fails, but breakouts still occur in strong moves).
# Designed for ~20-40 trades/year to minimize fee drag.

name = "4h_KC_Breakout_Squeeze_Volume_12hTrend"
timeframe = "4h"
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
    
    # Bollinger Bands for bandwidth (20, 2)
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
    tr[0] = tr1[0]
    
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
    
    # Bollinger Bandwidth regime: low volatility = squeeze (below 20th percentile)
    bbw_series = pd.Series(bb_width.values)
    bbw_rank = bbw_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    squeeze_condition = bbw_rank < 0.2  # Below 20th percentile = low volatility squeeze
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5  # Need enough history for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(squeeze_condition.iloc[i] if hasattr(squeeze_condition, 'iloc') else squeeze_condition[i]) or \
           np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Extract values safely
        sq_val = squeeze_condition.iloc[i] if hasattr(squeeze_condition, 'iloc') else squeeze_condition[i]
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper KC, in squeeze, with volume, and 12h trend up
            if close[i] > kc_upper[i] and sq_val and vol_confirm and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower KC, in squeeze, with volume, and 12h trend down
            elif close[i] < kc_lower[i] and sq_val and vol_confirm and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below middle KC OR squeeze ends (volatility expansion)
            if close[i] < kc_middle[i] or not sq_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above middle KC OR squeeze ends
            if close[i] > kc_middle[i] or not sq_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals