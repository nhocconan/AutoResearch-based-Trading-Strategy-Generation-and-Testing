#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_3ATR_Squeeze_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA100 for trend filter
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 1d ATR for volatility filter
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # 6h ATR for squeeze detection
    tr_6h = np.maximum(high - low, 
                       np.maximum(np.abs(high - np.roll(close, 1)), 
                                  np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr10_6h = pd.Series(tr_6h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Squeeze condition: 6h ATR < 3 * 1d ATR (low volatility breakout setup)
    squeeze = atr10_6h < (3 * atr14_1d_aligned)
    
    # Donchian breakout levels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 6h volume > 30-period average
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA100 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema100_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or
            np.isnan(vol_ma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian high during low volatility squeeze, with uptrend filter
            long_cond = (close[i] > donch_high[i] and 
                        squeeze[i] and
                        close[i] > ema100_1d_aligned[i] and
                        volume[i] > vol_ma30[i])
            
            # Short: Breakdown below Donchian low during low volatility squeeze, with downtrend filter
            short_cond = (close[i] < donch_low[i] and 
                         squeeze[i] and
                         close[i] < ema100_1d_aligned[i] and
                         volume[i] > vol_ma30[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below Donchian low OR trend reverses
            if close[i] < donch_low[i] or close[i] < ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above Donchian high OR trend reverses
            if close[i] > donch_high[i] or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Low volatility breakout strategy using 6-hour ATR squeeze (< 3x daily ATR) combined with Donchian(20) breakouts.
# Works in both bull and bear markets: squeeze identifies compression before explosive moves, trend filter ensures 
# alignment with higher timeframe direction. Volume confirmation avoids false breakouts. Targets 15-35 trades/year.