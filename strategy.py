#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Breakout with 4h Trend and Volume Confirmation
# Long when: Price breaks above Camarilla R3 level AND 4h close > 4h EMA50 AND 1h volume > 1.5x 20-period average volume
# Short when: Price breaks below Camarilla S3 level AND 4h close < 4h EMA50 AND 1h volume > 1.5x 20-period average volume
# Exit when price returns to Camarilla Pivot Point (mean reversion)
# Camarilla levels provide high-probability intraday support/resistance
# 4h EMA50 filter ensures we trade with the higher timeframe trend
# Volume confirmation ensures breakouts have conviction
# Target: 80-120 total trades over 4 years (20-30/year) with discrete sizing 0.20
# Works in both bull and bear markets by combining HTF trend direction with LTf precision entries

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Need sufficient lookback for Camarilla calculation (previous day's range)
        if i < 24:  # Need at least 24 hours of data for daily pivot
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using previous day's high, low, close
        # Find index of previous day's bar (assuming 24 1h bars per day)
        prev_day_idx = i - 24
        if prev_day_idx < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Previous day's OHLC
        prev_high = high[prev_day_idx:i].max()  # High of previous day
        prev_low = low[prev_day_idx:i].min()    # Low of previous day
        prev_close = close[prev_day_idx]        # Close of previous day (simplified)
        
        # Camarilla levels calculation
        range_val = prev_high - prev_low
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Camarilla levels
        pivot = (prev_high + prev_low + prev_close) / 3
        r3 = pivot + (range_val * 1.1 / 4)
        s3 = pivot - (range_val * 1.1 / 4)
        
        # Volume filter: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > (1.5 * vol_ma)
        else:
            volume_filter = False
        
        # 4h trend filter
        trend_filter_long = ema_50_4h_aligned[i] > 0 and close[i] > ema_50_4h_aligned[i]
        trend_filter_short = ema_50_4h_aligned[i] > 0 and close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: Break above R3 with 4h uptrend and volume confirmation
            if close[i] > r3 and trend_filter_long and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below S3 with 4h downtrend and volume confirmation
            elif close[i] < s3 and trend_filter_short and volume_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: return to pivot point (mean reversion)
            if close[i] < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: return to pivot point (mean reversion)
            if close[i] > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals