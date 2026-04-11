#!/usr/bin/env python3
# 4h_12h_camarilla_volume_trend_v1
# Strategy: 4h Camarilla pivot breakout with 12h EMA trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels from 12h provide institutional support/resistance.
# Breakouts above/below H3/L3 with 12h EMA trend alignment and volume > 1.5x 20-period average
# capture institutional moves. Works in bull markets via trend continuation and bear markets
# via short signals during distribution. Low trade frequency (~25-40/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_trend_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar (H3, L3)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # We use H3 and L3 as key levels
    range_12h = df_12h['high'] - df_12h['low']
    camarilla_h3 = df_12h['close'] + 1.1 * range_12h * 1.1 / 4
    camarilla_l3 = df_12h['close'] - 1.1 * range_12h * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for 12h bar to close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3.values)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h3_aligned[i]
        breakout_down = close[i] < camarilla_l3_aligned[i]
        
        # Trend filter: price above/below 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions
        # Long: breakout above H3 AND uptrend AND volume confirmation
        if breakout_up and trend_up and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: breakout below L3 AND downtrend AND volume confirmation
        elif breakout_down and trend_down and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to opposite Camarilla level (mean reversion)
        elif position == 1 and close[i] < camarilla_l3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > camarilla_h3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals