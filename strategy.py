#!/usr/bin/env python3
"""
Hypothesis: 12-hour Bollinger Band squeeze breakout with volume confirmation and 1-week trend filter.
Long when price breaks above upper BB after squeeze (BB width < 20th percentile) with volume > 1.5x average and weekly close > weekly EMA20.
Short when price breaks below lower BB after squeeze with volume > 1.5x average and weekly close < weekly EMA20.
Exit when price returns to middle BB or volatility expands (BB width > 80th percentile).
Designed for low-frequency, high-quality breakouts in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for Bollinger Bands - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour Bollinger Bands (20, 2)
    close_12h = df_12h['close'].values
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    middle_bb = sma_20
    bb_width = (upper_bb - lower_bb) / middle_bb * 100  # Percentage width
    
    # Calculate Bollinger Band width percentiles for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_20th = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20)
    bb_width_80th = bb_width_series.rolling(window=50, min_periods=20).quantile(0.80)
    
    # Calculate volume average
    volume_12h = df_12h['volume'].values
    volume_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean()
    
    # Calculate 1-week EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean()
    
    # Align all indicators to 12h timeframe (our base)
    sma_20_aligned = align_htf_to_ltf(df_12h, df_12h, sma_20.values)
    std_20_aligned = align_htf_to_ltf(df_12h, df_12h, std_20.values)
    upper_bb_aligned = align_htf_to_ltf(df_12h, df_12h, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(df_12h, df_12h, lower_bb.values)
    middle_bb_aligned = align_htf_to_ltf(df_12h, df_12h, middle_bb.values)
    bb_width_aligned = align_htf_to_ltf(df_12h, df_12h, bb_width.values)
    bb_width_20th_aligned = align_htf_to_ltf(df_12h, df_12h, bb_width_20th.values)
    bb_width_80th_aligned = align_htf_to_ltf(df_12h, df_12h, bb_width_80th.values)
    volume_ma_aligned = align_htf_to_ltf(df_12h, df_12h, volume_ma.values)
    ema_20_1w_aligned = align_htf_to_ltf(df_12h, df_1w, ema_20_1w.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(bb_width_aligned[i]) or np.isnan(bb_width_20th_aligned[i]) or 
            np.isnan(bb_width_80th_aligned[i]) or np.isnan(volume_ma_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bb_width_val = bb_width_aligned[i]
        bb_width_20th_val = bb_width_20th_aligned[i]
        bb_width_80th_val = bb_width_80th_aligned[i]
        volume_val = volume[i] if i < len(volume) else 0
        volume_ma_val = volume_ma_aligned[i]
        close_val = close[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        middle_bb_val = middle_bb_aligned[i]
        ema_20_1w_val = ema_20_1w_aligned[i]
        
        if position == 0:
            # Squeeze condition: BB width below 20th percentile (low volatility)
            is_squeeze = bb_width_val <= bb_width_20th_val
            # Volume confirmation: volume > 1.5x average
            volume_confirm = volume_val > 1.5 * volume_ma_val
            
            # Long: break above upper BB during squeeze with weekly uptrend
            if (is_squeeze and volume_confirm and 
                close_val > upper_bb_val and 
                close_12h[i] > upper_bb_val and  # Confirm on 12h close
                close_12h[i] > ema_20_1w_val):  # Weekly trend filter
                signals[i] = 0.25
                position = 1
            # Short: break below lower BB during squeeze with weekly downtrend
            elif (is_squeeze and volume_confirm and 
                  close_val < lower_bb_val and 
                  close_12h[i] < lower_bb_val and  # Confirm on 12h close
                  close_12h[i] < ema_20_1w_val):  # Weekly trend filter
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: return to middle BB or volatility expansion
                if (close_val <= middle_bb_val or 
                    bb_width_val >= bb_width_80th_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: return to middle BB or volatility expansion
                if (close_val >= middle_bb_val or 
                    bb_width_val >= bb_width_80th_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_BB_Squeeze_Breakout_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0