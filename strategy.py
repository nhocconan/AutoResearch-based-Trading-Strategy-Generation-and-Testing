#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1w Trend Filter and Volume Confirmation
# Long when: BB Width at 20-period low (squeeze) AND price breaks above upper BB AND 1w close > 1w EMA50 AND volume > 2x 20-period MA
# Short when: BB Width at 20-period low (squeeze) AND price breaks below lower BB AND 1w close < 1w EMA50 AND volume > 2x 20-period MA
# Exit when: price returns to middle BB OR volatility expands (BB Width > 1.5x 20-period MA of BB Width)
# Uses Bollinger Squeeze for low volatility breakouts, 1w EMA for trend filter, volume for conviction
# Timeframe: 6h, HTF: 1w for trend. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_BBSqueeze_1wTrend_VolumeBreakout"
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
    
    # Calculate Bollinger Bands on 6h (20, 2)
    if len(close) >= 20:
        ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = ma_20 + (2 * std_20)
        lower_bb = ma_20 - (2 * std_20)
        middle_bb = ma_20
        bb_width = upper_bb - lower_bb
    else:
        ma_20 = np.full(n, np.nan)
        std_20 = np.full(n, np.nan)
        upper_bb = np.full(n, np.nan)
        lower_bb = np.full(n, np.nan)
        middle_bb = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # Bollinger Squeeze: BB Width at 20-period low
    if len(bb_width) >= 20:
        bb_width_ma_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
        squeeze_condition = bb_width < bb_width_ma_20  # Width below its MA = squeeze
    else:
        squeeze_condition = np.zeros(n, dtype=bool)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w
    close_1w = df_1w['close'].values
    if len(close_1w) >= 50:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50_1w = np.full(len(df_1w), np.nan)
    
    # 1w trend: close > EMA50 = uptrend, close < EMA50 = downtrend
    trend_up = close_1w > ema_50_1w
    trend_down = close_1w < ema_50_1w
    
    # Align 1w trend to 6h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after BB calculation warmup
        # Skip if any value is NaN
        if (np.isnan(squeeze_condition[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(middle_bb[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: squeeze + breakout above upper BB + 1w uptrend + volume
            if (squeeze_condition[i] and 
                close[i] > upper_bb[i] and 
                trend_up_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: squeeze + breakout below lower BB + 1w downtrend + volume
            elif (squeeze_condition[i] and 
                  close[i] < lower_bb[i] and 
                  trend_down_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB OR volatility expands (squeeze ends)
            if (close[i] < middle_bb[i] or not squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB OR volatility expands (squeeze ends)
            if (close[i] > middle_bb[i] or not squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals