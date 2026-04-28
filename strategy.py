#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA34 trend filter and volume confirmation.
# Uses Alligator jaws/teeth/lips (SMAs of median price) to identify trend direction and strength.
# 1w EMA34 filters for higher timeframe trend alignment. Volume spike confirms momentum.
# Designed to work in both bull and bear markets by following the 1w trend while using Alligator for entry/exit.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

name = "12h_WilliamsAlligator_1wEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator on 12h timeframe
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Alligator lines: SMAs of median price
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period SMA
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period SMA
    
    # Align Alligator lines (no additional delay needed for SMAs)
    jaws_aligned = align_htf_to_ltf(prices, prices, jaws)  # self-align for same timeframe
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(jaws_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA34 direction
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Alligator conditions: lips > teeth > jaws = bullish alignment
        # lips < teeth < jaws = bearish alignment
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaws_aligned[i]
        
        # Entry conditions
        long_entry = price_above_ema and bullish_alignment and volume_spike[i]
        short_entry = price_below_ema and bearish_alignment and volume_spike[i]
        
        # Exit conditions: Alligator lines cross (trend weakening)
        long_exit = lips_aligned[i] < jaws_aligned[i]  # lips cross below jaws
        short_exit = lips_aligned[i] > jaws_aligned[i]  # lips cross above jaws
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals