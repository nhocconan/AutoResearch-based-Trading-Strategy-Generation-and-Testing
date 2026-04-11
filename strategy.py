#!/usr/bin/env python3
# 12h_1w_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot reversal with 1w trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla levels provide high-probability reversal zones. 1w EMA ensures alignment with long-term trend. Volume > 2x 20-period average confirms institutional interest.
# Designed for low trade frequency (~15-30/year) to minimize fee drain. Works in bull markets via pullbacks to support and bear markets via bounces from resistance.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (H, L, C)
    # Standard Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # We use H3/L3 for tighter levels: H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    # These levels act as support/resistance with high reversal probability
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla H3 and L3 levels for each 1d bar
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (higher threshold for fewer trades)
        vol_confirm = volume[i] > 2.0 * vol_avg_20[i]
        
        # Trend filter: price above/below 1w EMA50
        trend_bullish = close[i] > ema_50_1w_aligned[i]
        trend_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        # Long: Price touches or crosses above L3 (support) AND bullish long-term trend AND volume confirmation
        if low[i] <= camarilla_l3_aligned[i] and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price touches or crosses below H3 (resistance) AND bearish long-term trend AND volume confirmation
        elif high[i] >= camarilla_h3_aligned[i] and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price moves back to the opposite Camarilla level or crosses EMA
        elif position == 1 and (high[i] >= camarilla_h3_aligned[i] or close[i] < ema_50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (low[i] <= camarilla_l3_aligned[i] or close[i] > ema_50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals