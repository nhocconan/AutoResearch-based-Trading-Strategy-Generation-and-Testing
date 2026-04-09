#!/usr/bin/env python3
# 4h_bollinger_breakout_volume_1dtrend_v1
# Hypothesis: 4h Bollinger Band breakouts with volume confirmation and 1d trend filter work in both bull and bear markets.
# Uses Bollinger Bands (20,2) - breakout above upper band for long, below lower band for short.
# Requires volume > 1.5x 20-period average and 1d EMA50 trend alignment.
# Target: 20-40 trades/year (80-160 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bollinger_breakout_volume_1dtrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = (bb_middle + 2 * bb_std).values
    bb_lower = (bb_middle - 2 * bb_std).values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below middle band or 1d trend turns bearish
            if close[i] < bb_middle.iloc[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above middle band or 1d trend turns bullish
            if close[i] > bb_middle.iloc[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper band with volume confirmation and 1d uptrend
            if close[i] > bb_upper[i] and volume[i] > vol_threshold[i] and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band with volume confirmation and 1d downtrend
            elif close[i] < bb_lower[i] and volume[i] > vol_threshold[i] and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals