#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Camarilla pivot levels with volume confirmation and 1-week trend filter
# Camarilla levels (H3/L3) from daily provide strong support/resistance for 12h timeframe
# Break above H3 or below L3 with volume > 2x 20-period average indicates institutional breakout
# Trend filter: 20-period EMA on 1-week timeframe to ensure alignment with major trend
# Works in bull/bear markets: breakouts capture trends, reversals within range capture pullbacks
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_Camarilla_H3L3_VolumeWeekTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla pivot point calculation
    # Pivot = (previous high + previous low + previous close) / 3
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Camarilla levels: H3/L3 (strongest resistance/support)
    h3 = prev_close + (range_ * 1.1)
    l3 = prev_close - (range_ * 1.1)
    
    # Align daily levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: >2x 20-period average (high threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Trend filter: 20-period EMA on 1-week timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    uptrend = close > ema_20_1w_aligned
    downtrend = close < ema_20_1w_aligned
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above H3 with volume confirmation and uptrend
            if close[i] > h3_aligned[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below L3 with volume confirmation and downtrend
            elif close[i] < l3_aligned[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below pivot (failed support) or reaches 2x risk (take profit)
            if close[i] < pivot or close[i] > h3_aligned[i] + (h3_aligned[i] - pivot):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above pivot (failed resistance) or reaches 2x risk (take profit)
            if close[i] > pivot or close[i] < l3_aligned[i] - (pivot - l3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals