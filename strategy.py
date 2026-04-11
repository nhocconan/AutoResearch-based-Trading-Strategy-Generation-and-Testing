#!/usr/bin/env python3
"""
6h_1d_kama_volume_regime_v1
Strategy: 6h KAMA trend with volume confirmation and 1d volatility regime filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: KAMA adapts to market noise, reducing whipsaws in chop. Combined with volume confirmation and low-volatility regime (using 1d Bollinger Band width percentile), it captures sustainable trends while avoiding false signals in high noise. Designed to work in both bull (trend following) and bear (mean reversion in low vol) markets by switching logic based on volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_kama_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 6h KAMA (adaptive moving average)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of absolute changes
    # Handle edge cases for convolution-like sum
    volatility_padded = np.concatenate([np.zeros(9), volatility])
    volatility_sum = np.convolve(volatility_padded, np.ones(10), mode='valid')
    # Avoid division by zero
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d Bollinger Band width for volatility regime
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_width = (bb_std * 2) / bb_middle  # Normalized width
    # Percentile rank of BB width over 60 days
    bb_width_series = pd.Series(bb_width)
    bb_percentile = bb_width_series.rolling(window=60, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    # Align BB percentile to 6h
    bb_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_percentile)
    
    # Volume confirmation (20-period average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(bb_percentile_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        kama_val = kama[i]
        bb_percentile_val = bb_percentile_aligned[i]
        vol_confirmed = vol_spike[i]
        
        # Regime filter: low volatility (trending) vs high volatility (choppy)
        # Low vol: BB percentile < 40% (narrow bands = trending)
        # High vol: BB percentile > 60% (wide bands = choppy)
        low_vol = bb_percentile_val < 0.4
        high_vol = bb_percentile_val > 0.6
        
        # In low volatility regime: trend following with KAMA
        # In high volatility regime: mean reversion (fade extreme moves)
        if low_vol:
            # Trend following: price > KAMA = long, price < KAMA = short
            long_signal = price_close > kama_val and vol_confirmed
            short_signal = price_close < kama_val and vol_confirmed
        elif high_vol:
            # Mean reversion: fade moves away from KAMA
            # Long when price significantly below KAMA
            # Short when price significantly above KAMA
            deviation = (price_close - kama_val) / kama_val
            long_signal = deviation < -0.015 and vol_confirmed  # >1.5% below KAMA
            short_signal = deviation > 0.015 and vol_confirmed   # >1.5% above KAMA
        else:
            # Neutral regime: no trades
            long_signal = False
            short_signal = False
        
        # Exit conditions
        exit_long = position == 1 and price_close < kama_val
        exit_short = position == -1 and price_close > kama_val
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals