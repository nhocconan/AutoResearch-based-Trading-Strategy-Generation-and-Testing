#!/usr/bin/env python3
"""
6h_HeikinAshi_1dTrend_Volume
Hypothesis: Use Heikin-Ashi candles on 6h for smooth trend visualization, filtered by 1d EMA50 trend direction and volume spikes.
Heikin-Ashi reduces noise and highlights true trend direction, ideal for 6h timeframe to avoid whipsaws.
In bull markets: go long when HA closes above opens with 1d uptrend and volume spike.
In bear markets: go short when HA closes below opens with 1d downtrend and volume spike.
Volume confirmation ensures breakouts have participation, reducing false signals.
Designed for low trade frequency (~20-50/year) with high win rate by requiring confluence.
"""

name = "6h_HeikinAshi_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate Heikin-Ashi close and open
    ha_close = (high + low + close + close) / 4  # Using current close as approximation for prior close in HA calc
    # For HA open: ( prior HA open + prior HA close ) / 2
    # We'll compute iteratively but vectorized approximation: use prior close for simplicity in trend detection
    # More accurate: compute ha_open recursively, but for trend direction, ha_close - ha_open ~ close - prior_close smoothed
    # Instead, use: ha_open = (previous ha_open + previous ha_close)/2 -> complex in vector
    # Alternative: use actual HA calculation via loop for clarity (only 50-100 iterations needed for warmup)
    ha_open = np.zeros(n)
    ha_close_calc = np.zeros(n)
    ha_open[0] = (open_price := prices['open'].iloc[0])  # seed
    ha_close_calc[0] = (high[0] + low[0] + close[0] + open_price) / 4
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close_calc[i-1]) / 2
        ha_close_calc[i] = (high[i] + low[i] + close[i] + ha_open[i]) / 4
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 2x 20-period average volume (strict for fewer trades)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and HA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_6h[i]) or np.isnan(ha_open[i]) or np.isnan(ha_close_calc[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Heikin-Ashi trend: bullish when close > open, bearish when close < open
        ha_bullish = ha_close_calc[i] > ha_open[i]
        ha_bearish = ha_close_calc[i] < ha_open[i]
        
        trend_up = close[i] > ema_50_6h[i]
        trend_down = close[i] < ema_50_6h[i]
        
        if position == 0:
            # Long: HA bullish + 1d uptrend + volume spike
            if ha_bullish and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: HA bearish + 1d downtrend + volume spike
            elif ha_bearish and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: HA bearish or trend reversal
            if ha_bearish or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: HA bullish or trend reversal
            if ha_bullish or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals