#!/usr/bin/env python3
# 4h_VWAP_Reversion_Squeeze_12hTrend
# Hypothesis: Mean reversion from VWAP with Bollinger Band squeeze filter and 12h trend alignment.
# Works in both bull/bear markets: VWAP provides dynamic mean, Bollinger squeeze filters low volatility breakouts,
# and 12h trend ensures trades align with higher timeframe momentum. Target: 20-40 trades/year.

name = "4h_VWAP_Reversion_Squeeze_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP calculation (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Bollinger Bands (20-period, 2 std)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # % bandwidth
    
    # Bollinger squeeze: low volatility condition (20th percentile of bandwidth)
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < bb_width_ma  # volatility below average
    
    # 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(vwap[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(squeeze_condition[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below VWAP, not in squeeze, 12h uptrend
            if (close[i] < vwap[i] and
                not squeeze_condition[i] and
                trend_12h_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP, not in squeeze, 12h downtrend
            elif (close[i] > vwap[i] and
                  not squeeze_condition[i] and
                  trend_12h_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses above VWAP or 12h trend turns down
            if (close[i] > vwap[i] or
                trend_12h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses below VWAP or 12h trend turns up
            if (close[i] < vwap[i] or
                trend_12h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals