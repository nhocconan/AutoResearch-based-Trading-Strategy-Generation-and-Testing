#!/usr/bin/env python3
# 4h_1d_Camarilla_R1_S1_Breakout_With_Volume
# Hypothesis: Camarilla R1/S1 levels from daily chart act as strong support/resistance.
# Breakout above R1 or below S1 with volume confirmation and trend filter (1d EMA34) captures momentum.
# Designed for low trade frequency (20-40/year) to minimize fee drag in bull and bear markets.
# Works in bull via breakouts above R1 in uptrend, and in bear via breakdowns below S1 in downtrend.

name = "4h_1d_Camarilla_R1_S1_Breakout_With_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels: (H+L+C)/3 +/- (H-L)*1.1/2
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    r1_level = camarilla_pivot + (camarilla_range * 1.1 / 2)
    s1_level = camarilla_pivot - (camarilla_range * 1.1 / 2)
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    r1_prev = np.roll(r1_level, 1)
    s1_prev = np.roll(s1_level, 1)
    r1_prev[0] = r1_level[0]  # first day uses same day's level
    s1_prev[0] = s1_level[0]
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_prev)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slope_1d = np.diff(ema_34_1d, prepend=ema_34_1d[0])  # slope = today - yesterday
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    # ATR for volatility and trailing stop
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filter from 1d EMA34 slope
        bullish_trend = ema_slope_1d_aligned[i] > 0
        bearish_trend = ema_slope_1d_aligned[i] < 0
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: break above R1 in bullish trend with volume surge
            if close[i] > r1_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: break below S1 in bearish trend with volume surge
            elif close[i] < s1_aligned[i] and bearish_trend and volume_surge:
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