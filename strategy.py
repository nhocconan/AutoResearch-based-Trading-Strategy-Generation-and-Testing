#!/usr/bin/env python3
# 12h_1w_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: 12h breakout above/below weekly Camarilla R1/S1 with volume confirmation and weekly trend filter.
# Uses weekly EMA40 for trend bias to work in both bull and bear markets. Designed for low trade frequency.
# Weekly timeframe reduces noise and avoids overtrading on lower timeframes.

name = "12h_1w_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA40 for trend filter
    close_1w = df_1w['close'].values
    ema_40 = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_aligned = align_htf_to_ltf(prices, df_1w, ema_40)
    
    # Weekly high, low, close for Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    r1 = close_1w + 1.1 * (high_1w - low_1w) / 12
    s1 = close_1w - 1.1 * (high_1w - low_1w) / 12
    
    # Align R1 and S1 to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # ATR for volatility and trailing stop
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (2.5x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_40_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filter from weekly EMA40
        bullish_trend = close[i] > ema_40_aligned[i]
        bearish_trend = close[i] < ema_40_aligned[i]
        
        # Volume confirmation (2.5x average)
        volume_surge = volume[i] > 2.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above R1 in bullish trend with volume surge
            if close[i] > r1_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: breakdown below S1 in bearish trend with volume surge
            elif close[i] < s1_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        else:
            if position == 1:
                # Update highest high since entry
                if high[i] > highest_high_since_entry:
                    highest_high_since_entry = high[i]
                
                # Trailing stop: exit if price drops 2.0*ATR from highest high
                if close[i] < highest_high_since_entry - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Update lowest low since entry
                if low[i] < lowest_low_since_entry:
                    lowest_low_since_entry = low[i]
                
                # Trailing stop: exit if price rises 2.0*ATR from lowest low
                if close[i] > lowest_low_since_entry + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals