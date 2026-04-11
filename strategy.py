#!/usr/bin/env python3
# 12h_1d_camarilla_volume_v1
# Strategy: 12h Camarilla pivot breakout with volume confirmation and 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla levels from prior 1d act as strong support/resistance. 
# Long when price breaks above H3 with volume > 1.5x 20-period average and 1d close > open (bullish bias).
# Short when price breaks below L3 with volume confirmation and 1d close < open (bearish bias).
# Uses 1d trend filter to avoid counter-trend trades. Designed for low trade frequency (~15-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d (H4, L3, H3, L4 etc.)
    # Using prior day's high, low, close
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Camarilla multipliers
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    rang = phigh - plow
    h3 = pclose + 1.1 * rang
    l3 = pclose - 1.1 * rang
    h4 = pclose + 1.5 * rang
    l4 = pclose - 1.5 * rang
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 1d trend filter: bullish if close > open, bearish if close < open
    trend_bull = pclose > df_1d['open'].values
    trend_bear = pclose < df_1d['open'].values
    trend_bull_aligned = align_htf_to_ltf(prices, df_1d, trend_bull)
    trend_bear_aligned = align_htf_to_ltf(prices, df_1d, trend_bear)
    
    # Volume confirmation: current 12h volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or \
           np.isnan(trend_bull_aligned[i]) or np.isnan(trend_bear_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        # Long: price breaks above H3 AND 1d bullish trend AND volume confirmation
        if close[i] > h3_aligned[i] and trend_bull_aligned[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price breaks below L3 AND 1d bearish trend AND volume confirmation
        elif close[i] < l3_aligned[i] and trend_bear_aligned[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to opposite Camarilla level (mean reversion)
        elif position == 1 and close[i] < l3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > h3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals