#!/usr/bin/env python3
"""
6h_ThreeBarReversal_12hTrend_Filter
Hypothesis: Three-bar reversal pattern (bullish/bearish) on 6h chart with 12h EMA50 trend filter and volume confirmation.
Works in bull/bear markets by only taking reversals aligned with higher timeframe trend.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
"""

name = "6h_ThreeBarReversal_12hTrend_Filter"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.3 * vol_ma_20)
    
    # Three-bar reversal detection
    # Bullish: low < previous low AND close > previous close (after 2 down bars)
    # Bearish: high > previous high AND close < previous close (after 2 up bars)
    bullish_reversal = np.zeros(n, dtype=bool)
    bearish_reversal = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        # Bullish reversal: two consecutive lower lows followed by higher close
        if (low[i-2] > low[i-1] and 
            low[i-1] > low[i] and 
            close[i] > close[i-1]):
            bullish_reversal[i] = True
        
        # Bearish reversal: two consecutive higher highs followed by lower close
        if (high[i-2] < high[i-1] and 
            high[i-1] < high[i] and 
            close[i] < close[i-1]):
            bearish_reversal[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # Prevent overtrading (approx 1.5 days)
    
    start_idx = max(20, 50)  # Warmup for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 12h trend direction
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_12h_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_12h_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: bullish reversal in 12h uptrend with volume filter
            if (bullish_reversal[i] and 
                trend_12h_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: bearish reversal in 12h downtrend with volume filter
            elif (bearish_reversal[i] and 
                  trend_12h_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit conditions: bearish reversal OR trend change
            if (bearish_reversal[i] or not trend_12h_up):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: bullish reversal OR trend change
            if (bullish_reversal[i] or not trend_12h_down):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals