#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Breakout_1dTrend_VolumeSqueeze"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = (close_1d > ema34_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 1d ATR for volatility regime (volatility squeeze detection)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - close_1d_prev)
    tr3 = np.abs(low_1d - close_1d_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Donchian channel (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volatility squeeze: current 4h ATR < 0.5 * 1d ATR (low volatility environment)
    tr_4h = np.maximum(np.abs(high - low), np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_4h[0] = np.abs(high[0] - low[0])
    atr10_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    vol_squeeze = atr10_4h < (0.5 * atr14_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_squeeze[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volatility squeeze and 1d uptrend
            long_cond = (close[i] > highest_20[i] and vol_squeeze[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price breaks below Donchian low with volatility squeeze and 1d downtrend
            short_cond = (close[i] < lowest_20[i] and vol_squeeze[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low (reversal signal)
            if close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high (reversal signal)
            if close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout strategy with volatility squeeze filter and 1d EMA34 trend filter on 4h timeframe.
# Enters long when price breaks above 20-period high with volatility squeeze (low ATR) and 1d uptrend.
# Enters short when price breaks below 20-period low with volatility squeeze and 1d downtrend.
# Exits on reversal breakouts through the opposite Donchian band.
# Volatility squeeze identifies low-volatility environments preceding breakouts.
# Targets 20-35 trades/year on 4h timeframe to avoid overtrading.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from compression).
# Uses discrete sizing (0.25) to minimize churn and works on BTC/ETH/SOL.