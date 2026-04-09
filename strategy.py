#/usr/bin/env python3
# 4h_bollinger_band_squeeze_breakout_v1
# Hypothesis: Bollinger Band squeeze (low volatility) followed by breakout with volume confirmation.
# Works in both bull and bear markets by capturing volatility expansion after contraction.
# Uses 1d trend filter to avoid counter-trend trades. Target: 20-50 trades/year (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bollinger_band_squeeze_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    
    # Basis (SMA)
    basis = np.zeros(n)
    sum_close = 0.0
    for i in range(n):
        sum_close += close[i]
        if i >= bb_length:
            sum_close -= close[i - bb_length]
        if i >= bb_length - 1:
            basis[i] = sum_close / bb_length
    
    # Standard deviation
    dev = np.zeros(n)
    sum_sq = 0.0
    for i in range(n):
        sum_sq += close[i] * close[i]
        if i >= bb_length:
            sum_sq -= close[i - bb_length] * close[i - bb_length]
        if i >= bb_length - 1:
            variance = sum_sq / bb_length - basis[i] * basis[i]
            dev[i] = np.sqrt(max(variance, 0))
    
    upper = basis + bb_mult * dev
    lower = basis - bb_mult * dev
    
    # Bollinger Band Width (normalized)
    bb_width = (upper - lower) / basis
    bb_width = np.where(basis != 0, bb_width, 0)
    
    # Squeeze condition: BB Width below 20-period lower Bollinger Band of BB Width
    bb_width_ma = np.zeros(n)
    bb_width_std = np.zeros(n)
    sum_bbw = 0.0
    sum_bbw_sq = 0.0
    for i in range(n):
        sum_bbw += bb_width[i]
        sum_bbw_sq += bb_width[i] * bb_width[i]
        if i >= 20:
            sum_bbw -= bb_width[i - 20]
            sum_bbw_sq -= bb_width[i - 20] * bb_width[i - 20]
        if i >= 19:
            bb_width_ma[i] = sum_bbw / 20
            variance = sum_bbw_sq / 20 - bb_width_ma[i] * bb_width_ma[i]
            bb_width_std[i] = np.sqrt(max(variance, 0))
    
    # Lower Bollinger Band of BB Width
    bb_width_lower = bb_width_ma - 2.0 * bb_width_std
    squeeze = bb_width < bb_width_lower
    
    # Volume filter: 20-period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i - 20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    alpha_1d = 2 / (50 + 1)
    ema50_1d = np.zeros(len(df_1d))
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(df_1d)):
        ema50_1d[i] = alpha_1d * close_1d[i] + (1 - alpha_1d) * ema50_1d[i-1]
    
    trend_1d = np.where(close_1d > ema50_1d, 1, -1)
    trend_4h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(trend_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price touches lower band or trend turns bearish
            if close[i] < lower[i] or trend_4h[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper band or trend turns bullish
            if close[i] > upper[i] or trend_4h[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: break above upper band with volume and bullish trend (after squeeze)
            if (close[i] > upper[i] and 
                vol_ok and 
                trend_4h[i] == 1 and
                squeeze[i-1]):  # Was squeezed in previous bar
                position = 1
                signals[i] = 0.25
            # Enter short: break below lower band with volume and bearish trend (after squeeze)
            elif (close[i] < lower[i] and 
                  vol_ok and 
                  trend_4h[i] == -1 and
                  squeeze[i-1]):  # Was squeezed in previous bar
                position = -1
                signals[i] = -0.25
    
    return signals