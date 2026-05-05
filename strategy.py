#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1w trend filter and volume confirmation
# Long when: price breaks above upper BB(20,2) + 1w close > 1w EMA50 + volume > 2.0x 24-period MA
# Short when: price breaks below lower BB(20,2) + 1w close < 1w EMA50 + volume > 2.0x 24-period MA
# Exit when: price returns to middle BB(20) or volume drops below average
# Uses Bollinger Bands for volatility breakouts, weekly EMA for trend filter, volume for conviction
# Timeframe: 6h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_BollingerBreakout_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands on 6h (20-period, 2 std dev)
    if len(close) >= 20:
        bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
    else:
        bb_middle = np.full(n, np.nan)
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
    
    # Volume confirmation: 24-period MA (equivalent to 1d lookback in 6h)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (2.0 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on 1w for trend filter
    if len(close_1w) >= 50:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        # 1w trend: close > EMA50 = uptrend, close < EMA50 = downtrend
        trend_up = close_1w > ema_50_1w
        trend_down = close_1w < ema_50_1w
    else:
        trend_up = np.zeros(len(close_1w), dtype=bool)
        trend_down = np.zeros(len(close_1w), dtype=bool)
        ema_50_1w = np.full(len(close_1w), np.nan)
    
    # Align 1w trend to 6h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(volume_filter[i]) or 
            (i < len(trend_up_aligned) and (np.isnan(trend_up_aligned[i]) if hasattr(trend_up_aligned[i], '__iter__') else False)) or
            (i < len(trend_down_aligned) and (np.isnan(trend_down_aligned[i]) if hasattr(trend_down_aligned[i], '__iter__') else False))):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ensure we don't go out of bounds on aligned arrays
        if i >= len(trend_up_aligned) or i >= len(trend_down_aligned):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        if position == 0:
            # Long conditions: price breaks above upper BB + 1w uptrend + volume filter
            if (close[i] > bb_upper[i] and 
                trend_up_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower BB + 1w downtrend + volume filter
            elif (close[i] < bb_lower[i] and 
                  trend_down_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB or volume drops below average
            if (close[i] < bb_middle[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB or volume drops below average
            if (close[i] > bb_middle[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals