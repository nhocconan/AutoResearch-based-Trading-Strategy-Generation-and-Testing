#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w MACD histogram for trend filter, 1d Bollinger Band breakout, and volume confirmation.
# Long when 1w MACD histogram > 0, price breaks above BB upper band, volume > 1.5x average.
# Short when 1w MACD histogram < 0, price breaks below BB lower band, volume > 1.5x average.
# Bollinger Bands use 20-period SMA and 2 standard deviations.
# Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity and fee drag.
# Works in bull (trend follow) and bear (trend still exists in downtrends).

name = "1d_1wMACD_BB_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for MACD trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1w MACD(12,26,9)
    ema12 = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    macd_hist_pos = macd_hist > 0
    
    # 1d Bollinger Bands(20,2)
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std
    
    # Align 1w MACD histogram to 1d
    macd_hist_pos_aligned = align_htf_to_ltf(prices, df_1w, macd_hist_pos.astype(float))
    # Align Bollinger Bands to 1d
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(macd_hist_pos_aligned[i]) or np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1w MACD histogram > 0, price breaks above BB upper band, volume spike
            if (macd_hist_pos_aligned[i] and
                close[i] > bb_upper_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: 1w MACD histogram < 0, price breaks below BB lower band, volume spike
            elif (not macd_hist_pos_aligned[i] and
                  close[i] < bb_lower_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: MACD flip, price breaks below BB lower band, or max 20 days held
            if (not macd_hist_pos_aligned[i] or 
                close[i] < bb_lower_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: MACD flip, price breaks above BB upper band, or max 20 days held
            if (macd_hist_pos_aligned[i] or 
                close[i] > bb_upper_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals