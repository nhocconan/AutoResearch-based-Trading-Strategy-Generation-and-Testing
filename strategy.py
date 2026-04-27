#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Bollinger Band squeeze identifies low volatility periods; breakout from squeeze with
# volume and higher timeframe trend captures momentum moves in both bull and bear markets.
# Uses Bollinger Bands (20,2) for squeeze detection and breakout signals.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Bollinger Bands (20,2) on 4h
    bb_length = 20
    bb_mult = 2.0
    bb_mid = np.full(n, np.nan)
    bb_up = np.full(n, np.nan)
    bb_dn = np.full(n, np.nan)
    bb_width = np.full(n, np.nan)
    
    for i in range(bb_length - 1, n):
        ma = np.mean(close[i - bb_length + 1:i + 1])
        sigma = np.std(close[i - bb_length + 1:i + 1])
        bb_mid[i] = ma
        bb_up[i] = ma + bb_mult * sigma
        bb_dn[i] = ma - bb_mult * sigma
        bb_width[i] = bb_up[i] - bb_dn[i]
    
    # Bollinger Band width percentile for squeeze detection (50-period)
    bb_width_pct = np.full(n, np.nan)
    bb_width_len = 50
    for i in range(bb_width_len - 1, n):
        window = bb_width[i - bb_width_len + 1:i + 1]
        if np.all(~np.isnan(window)):
            current = bb_width[i]
            rank = np.sum(window < current) / bb_width_len
            bb_width_pct[i] = rank * 100
    
    # 1d EMA trend filter (50-period)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need BB (20), BB width percentile (50), EMA (50), volume MA (20)
    start_idx = max(bb_length, bb_width_len, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bb_up[i]) or np.isnan(bb_dn[i]) or 
            np.isnan(bb_width_pct[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Bollinger Band squeeze: width < 20th percentile (low volatility)
        squeeze = bb_width_pct[i] < 20
        
        # Breakout conditions
        breakout_up = price > bb_up[i]
        breakout_dn = price < bb_dn[i]
        
        # Volume filter: significant volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: breakout above upper band with squeeze, volume, and bullish trend
            if squeeze and breakout_up and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: breakout below lower band with squeeze, volume, and bearish trend
            elif squeeze and breakout_dn and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle band or trend turns bearish
            if price <= bb_mid[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle band or trend turns bullish
            if price >= bb_mid[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Bollinger_Squeeze_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0