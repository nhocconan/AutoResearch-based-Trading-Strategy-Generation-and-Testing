#!/usr/bin/env python3
name = "12h_WickReversal_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1w = close_1w > ema34_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # Calculate wick ratio: (high - close) / (high - low) for rejection candles
    # High wick ratio indicates bearish rejection, low wick ratio indicates bullish rejection
    body_size = np.abs(close - open_)
    total_range = high - low
    # Avoid division by zero
    total_range = np.where(total_range == 0, 1e-10, total_range)
    
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    
    # Bullish rejection: long lower wick, small body
    bullish_rejection = (lower_wick / total_range > 0.6) & (body_size / total_range < 0.3)
    # Bearish rejection: long upper wick, small body
    bearish_rejection = (upper_wick / total_range > 0.6) & (body_size / total_range < 0.3)
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > 2.0 * vol_ma24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trend_up_1w_aligned[i]) or np.isnan(vol_ma24[i]) or
            np.isnan(bullish_rejection[i]) or np.isnan(bearish_rejection[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish rejection + weekly uptrend + volume spike
            if bullish_rejection[i] and trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish rejection + weekly downtrend + volume spike
            elif bearish_rejection[i] and not trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish rejection OR weekly trend turns down
            if bearish_rejection[i] or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish rejection OR weekly trend turns up
            if bullish_rejection[i] or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals