#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below lower BB(20,2) AND BB width < 20th percentile (squeeze) AND close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price returns to middle BB (mean reversion) or BB width expands above 50th percentile (squeeze end)
# Bollinger squeeze captures low volatility contractions that precede explosive moves, effective in both bull and bear markets.
# Timeframe: 6h, HTF: 1d. Target: 60-140 total trades over 4 years (15-35/year).

name = "6h_BollingerSqueeze_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands on 6h
    if len(close) >= 20:
        ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = ma_20 + (2 * std_20)
        lower_bb = ma_20 - (2 * std_20)
        middle_bb = ma_20
        bb_width = (upper_bb - lower_bb) / ma_20  # Normalized width
    else:
        ma_20 = np.full(n, np.nan)
        std_20 = np.full(n, np.nan)
        upper_bb = np.full(n, np.nan)
        lower_bb = np.full(n, np.nan)
        middle_bb = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # Calculate volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate BB width percentiles for squeeze detection (using 50 lookback)
    bb_width_pct = np.full(n, np.nan)
    if len(bb_width) >= 50:
        for i in range(50, n):
            window = bb_width[i-50:i]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                current_val = bb_width[i]
                if not np.isnan(current_val):
                    percentile = (np.sum(valid_window <= current_val) / len(valid_window)) * 100
                    bb_width_pct[i] = percentile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ma_20[i]) or np.isnan(std_20[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(bb_width_pct[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper BB AND squeeze (BB width < 20th percentile) AND above 1d EMA50 AND volume filter
            if (close[i] > upper_bb[i] and 
                bb_width_pct[i] < 20 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB AND squeeze (BB width < 20th percentile) AND below 1d EMA50 AND volume filter
            elif (close[i] < lower_bb[i] and 
                  bb_width_pct[i] < 20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB OR squeeze ends (BB width > 50th percentile)
            if (close[i] <= middle_bb[i] or bb_width_pct[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB OR squeeze ends (BB width > 50th percentile)
            if (close[i] >= middle_bb[i] or bb_width_pct[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals