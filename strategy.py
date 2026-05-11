#!/usr/bin/env python3
# 4h_12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume surge.
# Works in bull via breakouts above R1 in uptrend, and in bear via breakdowns below S1 in downtrend.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get 12h data for additional trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    phigh[0] = high_1d[0]
    plow[0] = low_1d[0]
    pclose[0] = close_1d[0]
    
    # Camarilla calculations
    range_val = phigh - plow
    R1 = pclose + range_val * 1.1 / 12
    S1 = pclose - range_val * 1.1 / 12
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slope_1d = np.diff(ema_34_1d, prepend=ema_34_1d[0])
    
    # 12h EMA50 for additional trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope_12h = np.diff(ema_50_12h, prepend=ema_50_12h[0])
    
    # ATR for volatility and trailing stop
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    ema_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_slope_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(ema_slope_12h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filters: both 1d and 12h must agree
        bullish_trend = ema_slope_1d_aligned[i] > 0 and ema_slope_12h_aligned[i] > 0
        bearish_trend = ema_slope_1d_aligned[i] < 0 and ema_slope_12h_aligned[i] < 0
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: break above R1 in bullish trend with volume surge
            if close[i] > R1_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: break below S1 in bearish trend with volume surge
            elif close[i] < S1_aligned[i] and bearish_trend and volume_surge:
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