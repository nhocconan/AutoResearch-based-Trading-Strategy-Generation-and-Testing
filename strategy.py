#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Triple_Barrier_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = (close_1d > ema50_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 4h ATR for volatility breakout
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Dynamic breakout levels based on ATR
    upper_break = close + 0.5 * atr  # 0.5 ATR above close for long breakout
    lower_break = close - 0.5 * atr  # 0.5 ATR below close for short breakout
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for ATR and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma20[i]) or np.isnan(upper_break[i]) or np.isnan(lower_break[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper barrier with volume and 1d uptrend
            long_cond = (high[i] > upper_break[i] and vol_filter[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price breaks below lower barrier with volume and 1d downtrend
            short_cond = (low[i] < lower_break[i] and vol_filter[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below breakout level or trend reverses
            if close[i] < upper_break[i] or trend_1d_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above breakout level or trend reverses
            if close[i] > lower_break[i] or trend_1d_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Dynamic ATR-based breakout strategy with volume confirmation and 1d trend filter.
# Uses 0.5 * ATR(14) as breakout threshold to adapt to changing volatility.
# Volume filter requires 2x 20-period average to confirm breakout strength.
# Trend filter uses 1d EMA50 to ensure alignment with higher timeframe direction.
# Position size fixed at 0.25 to manage risk and reduce fee impact.
# Designed to work in both bull (breakouts with trend) and bear (breakouts against trend) markets.
# Target: 20-40 trades/year to minimize fee drag while capturing significant moves.