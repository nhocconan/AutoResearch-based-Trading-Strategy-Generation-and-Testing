#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze + 1d Volume Spike + 1w Trend Filter.
# Long when: BB width < 20th percentile (squeeze), price breaks above upper band, volume > 2x 20-period EMA, price > 1w EMA.
# Short when: BB width < 20th percentile (squeeze), price breaks below lower band, volume > 2x 20-period EMA, price < 1w EMA.
# Exit when: price crosses middle band (20-period SMA) or BB width > 50th percentile (squeeze ends).
# Designed to capture breakouts from low volatility periods with volume confirmation and trend alignment.
# Works in both bull and bear markets by following 1w EMA direction.
# Bollinger Squeeze identifies contraction before expansion, reducing whipsaw.
name = "4h_BollingerSqueeze_1dVolume_1wTrend"
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
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    sma = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    std = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper = sma + bb_mult * std
    lower = sma - bb_mult * std
    bb_width = upper - lower
    
    # Percentile lookback for squeeze detection (50 periods)
    def rolling_percentile(arr, window, percentile):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(len(arr)):
            if i < window - 1:
                continue
            window_data = arr[i-window+1:i+1]
            valid = window_data[~np.isnan(window_data)]
            if len(valid) >= window:
                result[i] = np.percentile(valid, percentile)
        return result
    
    bb_width_p20 = rolling_percentile(bb_width, 50, 20.0)
    bb_width_p50 = rolling_percentile(bb_width, 50, 50.0)
    squeeze = bb_width < bb_width_p20
    
    # Volume confirmation: volume > 2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    # 1w EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_length, 20)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(sma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(bb_width_p20[i]) or np.isnan(bb_width_p50[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: squeeze + break above upper + volume + 1w EMA up
            if squeeze[i] and price > upper[i] and vol_confirm[i] and price > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze + break below lower + volume + 1w EMA down
            elif squeeze[i] and price < lower[i] and vol_confirm[i] and price < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle band OR squeeze ends (width > p50)
            if price < sma[i] or bb_width[i] > bb_width_p50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle band OR squeeze ends
            if price > sma[i] or bb_width[i] > bb_width_p50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals