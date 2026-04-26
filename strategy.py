#!/usr/bin/env python3
"""
4h_KAMA_Direction_1dEMA34_Trend_VolumeSpike_ATRStop
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 4h for trend direction, filtered by 1d EMA34 trend and volume spike. Enter long when 4h price crosses above KAMA with 1d uptrend and volume spike; short when crosses below KAMA with 1d downtrend and volume spike. ATR-based stoploss manages risk. Designed to capture momentum with controlled trade frequency for BTC/ETH in both bull and bear markets.
"""

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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate KAMA on 4h (ER=10, fast=2, slow=30)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of absolute changes over 10 periods
    # Vectorized volatility calculation
    volatility = pd.Series(close).rolling(window=10, min_periods=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of KAMA seed(10), EMA(34), ATR(14), volume MA(20)
    start_idx = max(10, 34, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_1d_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price crosses above KAMA AND 1d trend up AND volume spike
            long_signal = (close_val > kama[i]) and (close[i-1] <= kama[i-1]) and trend_1d_up and vol_spike
            
            # Short: price crosses below KAMA AND 1d trend down AND volume spike
            short_signal = (close_val < kama[i]) and (close[i-1] >= kama[i-1]) and trend_1d_down and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down OR price hits ATR stoploss
            if (not trend_1d_up) or (close_val < entry_price - 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_1d_down) or (close_val > entry_price + 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Direction_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0