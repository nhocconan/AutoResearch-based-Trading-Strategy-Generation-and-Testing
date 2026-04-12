#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v1
# Weekly Camarilla levels for daily trend bias with daily breakout entries.
# Uses weekly H4/L4 as trend filter and daily H3/L3 for entries with volume confirmation.
# Weekly timeframe reduces noise, daily provides timely entries.
# Works in bull markets via weekly uptrend + daily breakout above H3.
# Works in bear markets via weekly downtrend + daily breakdown below L3.
# Volume confirmation filters breakouts, avoiding false signals.
# Target: 15-25 trades/year per symbol for low friction.
name = "1d_1w_camarilla_breakout_v1"
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
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous week
    high_prev_w = df_1w['high'].shift(1).values
    low_prev_w = df_1w['low'].shift(1).values
    close_prev_w = df_1w['close'].shift(1).values
    
    range_prev_w = high_prev_w - low_prev_w
    weekly_h4 = close_prev_w + range_prev_w * 1.1 / 2
    weekly_l4 = close_prev_w - range_prev_w * 1.1 / 2
    weekly_h3 = close_prev_w + range_prev_w * 1.1 / 4
    weekly_l3 = close_prev_w - range_prev_w * 1.1 / 4
    
    # Align weekly levels to daily timeframe (already delayed by 1 week due to shift)
    weekly_h4_d = align_htf_to_ltf(prices, df_1w, weekly_h4)
    weekly_l4_d = align_htf_to_ltf(prices, df_1w, weekly_l4)
    weekly_h3_d = align_htf_to_ltf(prices, df_1w, weekly_h3)
    weekly_l3_d = align_htf_to_ltf(prices, df_1w, weekly_l3)
    
    # Daily volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if weekly levels not ready
        if np.isnan(weekly_h4_d[i]) or np.isnan(weekly_l4_d[i]):
            signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly H3 with volume, in weekly uptrend (above weekly L4)
        if (close[i] > weekly_h3_d[i] and 
            vol_confirm[i] and 
            close[i] > weekly_l4_d[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly L3 with volume, in weekly downtrend (below weekly H4)
        elif (close[i] < weekly_l3_d[i] and 
              vol_confirm[i] and 
              close[i] < weekly_h4_d[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout or trend reversal
        elif (position == 1 and 
              (close[i] < weekly_l3_d[i] or close[i] < weekly_l4_d[i])):
            position = 0
            signals[i] = 0.0
        elif (position == -1 and 
              (close[i] > weekly_h3_d[i] or close[i] > weekly_h4_d[i])):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals