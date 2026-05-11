#!/usr/bin/env python3
# 1h_Camarilla_R3_S3_4hTrend_Volume
# Hypothesis: 1h breakout at Camarilla R3/S3 levels with 4h EMA34 trend filter and volume confirmation.
# Uses 4h EMA34 for trend direction, daily Camarilla for entry levels, and volume spike for confirmation.
# Designed for low trade frequency (15-37/year) to minimize fee drag while capturing strong moves.

name = "1h_Camarilla_R3_S3_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Camarilla levels from previous day ---
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    range_hl = prev_high - prev_low
    r3 = prev_close + range_hl * 1.1 / 2
    s3 = prev_close - range_hl * 1.1 / 2
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # --- 4h trend: EMA34 slope ---
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slope_34_4h = np.diff(ema_34_4h, prepend=ema_34_4h[0])
    ema_slope_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_slope_34_4h)
    
    # --- ATR for volatility and trailing stop ---
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # --- Volume confirmation (2.5x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # --- Session filter: 08-20 UTC ---
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup: ensure we have enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Skip if any critical values are NaN
        if (not in_session or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_slope_34_4h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filter from 4h EMA34 slope
        bullish_trend = ema_slope_34_4h_aligned[i] > 0
        bearish_trend = ema_slope_34_4h_aligned[i] < 0
        
        # Volume confirmation (2.5x average)
        volume_surge = volume[i] > 2.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 in bullish trend with volume surge
            if close[i] > r3_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below S3 in bearish trend with volume surge
            elif close[i] < s3_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = low[i]
        else:
            if position == 1:
                # Update highest high since entry
                if high[i] > highest_high_since_entry:
                    highest_high_since_entry = high[i]
                
                # Trailing stop: exit if price drops 3.0*ATR from highest high
                if close[i] < highest_high_since_entry - 3.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Update lowest low since entry
                if low[i] < lowest_low_since_entry:
                    lowest_low_since_entry = low[i]
                
                # Trailing stop: exit if price rises 3.0*ATR from lowest low
                if close[i] > lowest_low_since_entry + 3.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.20
    
    return signals