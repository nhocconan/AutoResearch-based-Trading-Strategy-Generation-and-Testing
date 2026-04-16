#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with weekly trend filter
# Long when: Bollinger Band width < 20th percentile (squeeze) AND price breaks above upper band AND weekly close > weekly open
# Short when: Bollinger Band width < 20th percentile (squeeze) AND price breaks below lower band AND weekly close < weekly open
# Volume confirmation: volume > 1.5x 6s average volume
# Exit: opposite band touch or volatility expansion (BB width > 80th percentile)
# This captures low-volatility breakouts in both bull and bear markets, with weekly trend filter to avoid counter-trend trades
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag while capturing explosive moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Bollinger Bands (20, 2) ===
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    bb_width = (upper - lower) / ma  # Normalized width
    
    # Bollinger Band width percentiles for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_20th = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20)
    bb_width_80th = bb_width_series.rolling(window=50, min_periods=50).quantile(0.80)
    
    # === 1w Trend Filter (weekly close vs open) ===
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # Bullish weekly candle
    weekly_bearish = weekly_close < weekly_open  # Bearish weekly candle
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # === 6s Volume Confirmation (average volume) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(bb_width_20th[i]) or np.isnan(bb_width_80th[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x 6s average volume
        vol_confirm = volume[i] > vol_ma_val * 1.5
        
        # Squeeze condition: BB width < 20th percentile (low volatility)
        squeeze = bb_width[i] < bb_width_20th[i]
        
        # Volatility expansion exit: BB width > 80th percentile
        volatility_expansion = bb_width[i] > bb_width_80th[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit if price touches lower band OR volatility expansion
            if price <= lower[i] or volatility_expansion:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit if price touches upper band OR volatility expansion
            if price >= upper[i] or volatility_expansion:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: squeeze AND price breaks above upper band AND weekly bullish AND volume confirmation
            if squeeze and price > upper[i] and weekly_bullish_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: squeeze AND price breaks below lower band AND weekly bearish AND volume confirmation
            elif squeeze and price < lower[i] and weekly_bearish_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_BollingerSqueeze_WeeklyTrend_Volume1.5x_ExitLowerUpper"
timeframe = "6h"
leverage = 1.0