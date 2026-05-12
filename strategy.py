#!/usr/bin/env python3
name = "6h_SqueezeBreakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # Load daily data for trend filter and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Bollinger Bands on 6h (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bandwidth = (upper - lower) / sma  # Bandwidth for squeeze detection
    
    # Bollinger Bandwidth squeeze: bandwidth < 50th percentile of last 50 periods
    bandwidth_series = pd.Series(bandwidth)
    bw_rank = bandwidth_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).values
    squeeze = bw_rank < 0.5  # True when in squeeze (low volatility)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(squeeze[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: squeeze breakout up + above daily EMA200 + volume spike
            if close[i] > upper[i] and squeeze[i-1] and close[i] > ema_200_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down + below daily EMA200 + volume spike
            elif close[i] < lower[i] and squeeze[i-1] and close[i] < ema_200_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below middle Bollinger Band (SMA)
            if close[i] < sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above middle Bollinger Band (SMA)
            if close[i] > sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals