#!/usr/bin/env python3
# 12h_1d_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: On 12h chart, buy when price breaks above previous day's Camarilla R1 level in a daily uptrend with volume surge,
# sell when price breaks below previous day's Camarilla S1 level in a daily downtrend with volume surge.
# Uses daily trend filter to avoid counter-trend trades, volume confirmation to ensure conviction,
# and Camarilla levels for precise intraday support/resistance. Designed for low trade frequency (12-37/year) to minimize fee drag.
# Works in bull via daily uptrend + R1 breakouts, and in bear via daily downtrend + S1 breakdowns.

name = "12h_1d_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_open = df_1d['open'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (no look-ahead)
    prev_open_1 = np.roll(prev_open, 1)
    prev_high_1 = np.roll(prev_high, 1)
    prev_low_1 = np.roll(prev_low, 1)
    prev_close_1 = np.roll(prev_close, 1)
    prev_open_1[0] = prev_open[0]
    prev_high_1[0] = prev_high[0]
    prev_low_1[0] = prev_low[0]
    prev_close_1[0] = prev_close[0]
    
    # Calculate Camarilla levels for previous day
    range_ = prev_high_1 - prev_low_1
    camarilla_r1 = prev_close_1 + range_ * 1.1 / 12
    camarilla_s1 = prev_close_1 - range_ * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily trend: 20-period EMA on daily close
    daily_ema20 = pd.Series(prev_close_1).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_ema20_aligned = align_htf_to_ltf(prices, df_1d, daily_ema20)
    
    # Daily volume average for confirmation
    daily_volume = df_1d['volume'].values
    daily_volume_ma = pd.Series(daily_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_volume_ma_aligned = align_htf_to_ltf(prices, df_1d, daily_volume_ma)
    
    # ATR for volatility and trailing stop (using 12h data)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(daily_ema20_aligned[i]) or
            np.isnan(daily_volume_ma_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filter from daily EMA20
        bullish_trend = close[i] > daily_ema20_aligned[i]
        bearish_trend = close[i] < daily_ema20_aligned[i]
        
        # Volume confirmation (1.5x daily average)
        volume_surge = volume[i] > 1.5 * daily_volume_ma_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in bullish trend with volume surge
            if close[i] > camarilla_r1_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below Camarilla S1 in bearish trend with volume surge
            elif close[i] < camarilla_s1_aligned[i] and bearish_trend and volume_surge:
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