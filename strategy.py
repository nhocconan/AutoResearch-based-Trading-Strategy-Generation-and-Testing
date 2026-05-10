#!/usr/bin/env python3
# 1d_1w_TRIX_Zero_Cross_Volume_Confirm
# Hypothesis: Daily TRIX (triple smoothed EMA) zero-cross for trend direction with 1-week EMA trend filter and volume confirmation.
# TRIX is effective in both trending and ranging markets; zero-cross indicates momentum shift.
# Combined with weekly EMA for higher timeframe trend bias and volume to confirm breakout strength.
# Designed for low trade frequency (<25/year) to minimize fee drag.

name = "1d_1w_TRIX_Zero_Cross_Volume_Confirm"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period triple EMA, then 1-period percent change)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema3).pct_change() * 100  # percent change
    trix_values = trix.values
    
    # Weekly EMA30 for trend filter
    close_1w = df_1w['close'].values
    ema30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # ATR for stoploss
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to daily timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_values)
    ema30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema30_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix_aligned[i]) or
            np.isnan(ema30_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # TRIX zero-cross for momentum
        trix_pos = trix_aligned[i] > 0
        trix_neg = trix_aligned[i] < 0
        
        # Weekly EMA trend filter
        bullish_trend = close[i] > ema30_1w_aligned[i]
        bearish_trend = close[i] < ema30_1w_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: TRIX positive in bullish weekly trend with volume surge
            if trix_pos and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: TRIX negative in bearish weekly trend with volume surge
            elif trix_neg and bearish_trend and volume_surge:
                signals[i] = -0.25
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
                    signals[i] = 0.25
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
                    signals[i] = -0.25
    
    return signals