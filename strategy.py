#!/usr/bin/env python3
# 1d_1w_4WeekLowBreakout_TrendFilter_Volume
# Hypothesis: 1d 4-week low breakout in direction of 1w EMA trend with volume confirmation.
# Captures momentum after prolonged consolidation, works in bull (continuation) and bear (rebound from deep value).
# Target: 15-25 trades/year to minimize fee drag.

name = "1d_1w_4WeekLowBreakout_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:  # need 4 weeks + buffer
        return np.zeros(n)
    
    # Get weekly data for trend filter and 4-week low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-week low (20-day low on 1d, but we use weekly close for robustness)
    close_1w = df_1w['close'].values
    low_4w = pd.Series(close_1w).rolling(window=4, min_periods=4).min().values  # 4-week low
    
    # 1w EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_slope_20_1w = np.diff(ema_20_1w, prepend=ema_20_1w[0])  # slope = today - yesterday
    
    # ATR for volatility and trailing stop
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 50-day average)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly indicators to daily
    low_4w_aligned = align_htf_to_ltf(prices, df_1w, low_4w)
    ema_slope_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup: need 4 weeks of weekly data + ATR/vol lookback
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(low_4w_aligned[i]) or
            np.isnan(ema_slope_20_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filter from 1w EMA20 slope
        bullish_trend = ema_slope_20_1w_aligned[i] > 0
        bearish_trend = ema_slope_20_1w_aligned[i] < 0
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: break above 4-week low in bullish trend with volume surge
            if close[i] > low_4w_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: break below 4-week low in bearish trend with volume surge
            elif close[i] < low_4w_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        else:
            if position == 1:
                # Update highest high since entry
                if high[i] > highest_high_since_entry:
                    highest_high_since_entry = high[i]
                
                # Trailing stop: exit if price drops 2.5*ATR from highest high
                if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Update lowest low since entry
                if low[i] < lowest_low_since_entry:
                    lowest_low_since_entry = low[i]
                
                # Trailing stop: exit if price rises 2.5*ATR from lowest low
                if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals