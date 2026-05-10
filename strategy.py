#!/usr/bin/env python3
# 1D_KAMA_Direction_1wTrend_Filter_Volume
# Hypothesis: Daily KAMA direction (trend) filtered by weekly trend and volume confirmation.
# KAMA adapts to market noise, reducing whipsaws in sideways markets. Weekly trend filter ensures
# alignment with higher-timeframe momentum. Volume confirmation adds conviction to breakouts.
# Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
# Works in bull markets by following weekly uptrend, in bear markets by following weekly downtrend.
# Uses discrete position sizing (0.25) to minimize churn.

name = "1D_KAMA_Direction_1wTrend_Filter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend direction
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate KAMA(10, 2, 30) on daily close
    # ER = |close - close[10]| / sum(|close - close[1]| over 10 periods)
    # SSC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prev_KAMA + SSC * (price - prev_KAMA)
    fast = 2
    slow = 30
    lookback = 10
    
    change = np.abs(np.subtract(close[lookback:], close[:-lookback]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # Vectorized volatility sum over lookback window
    volatility_sum = np.zeros_like(close)
    for i in range(lookback, len(close)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-lookback:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close)
    mask = volatility_sum != 0
    er[lookback:] = np.divide(change, volatility_sum[lookback:], out=np.zeros_like(change), where=mask[lookback:])
    
    sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Warmup for KAMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend alignment: price vs weekly EMA and KAMA direction
        price_above_weekly_ema = close[i] > ema_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_1w_aligned[i]
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        if position == 0:
            # Long entry: price above weekly EMA + KAMA rising + volume surge
            if (price_above_weekly_ema and 
                kama_rising and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below weekly EMA + KAMA falling + volume surge
            elif (price_below_weekly_ema and 
                  kama_falling and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below weekly EMA OR KAMA falls
            if (price_below_weekly_ema or not kama_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above weekly EMA OR KAMA rises
            if (price_above_weekly_ema or not kama_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals