#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_reversal_v1
# Strategy: 4h Camarilla pivot breakout with reversal signal filter (new approach)
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Breakouts at Camarilla H3/L3 levels are more reliable when preceded by
# a reversal candle (engulfing or pin bar) showing rejection of opposite level.
# This adds confluence while reducing false signals. Uses volume confirmation.
# Designed for 15-25 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_reversal_v1"
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
    open_ = prices['open'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    rng = prev_high - prev_low
    H3 = prev_close + 1.1 * rng / 4
    L3 = prev_close - 1.1 * rng / 4
    H4 = prev_close + 1.1 * rng / 2
    L4 = prev_close - 1.1 * rng / 2
    
    # Align Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Reversal candle detection: bullish engulfing or hammer
    body_size = np.abs(close - open_)
    lower_shadow = np.minimum(open_, close) - low
    upper_shadow = high - np.maximum(open_, close)
    
    # Bullish reversal: close > open AND (engulfing OR hammer)
    bullish_engulfing = (close > open_) & (open_ < np.roll(close, 1)) & (close > np.roll(open_, 1))
    bullish_hammer = (close > open_) & (lower_shadow > 2 * body_size) & (upper_shadow < 0.1 * body_size)
    bullish_reversal = bullish_engulfing | bullish_hammer
    
    # Bearish reversal: close < open AND (engulfing OR shooting star)
    bearish_engulfing = (close < open_) & (open_ > np.roll(close, 1)) & (close < np.roll(open_, 1))
    bearish_star = (close < open_) & (upper_shadow > 2 * body_size) & (lower_shadow < 0.1 * body_size)
    bearish_reversal = bearish_engulfing | bearish_star
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume[i] > 1.8 * vol_avg_20[i]
        
        # Breakout signals using Camarilla levels with reversal filter
        breakout_up = high[i] > H3_aligned[i-1]
        breakdown_down = low[i] < L3_aligned[i-1]
        
        # Entry conditions with reversal filter
        # Long: Breakout above H3 AND volume confirmation AND bullish reversal candle
        if breakout_up and vol_confirm and bullish_reversal[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below L3 AND volume confirmation AND bearish reversal candle
        elif breakdown_down and vol_confirm and bearish_reversal[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout using H4/L4 levels
        elif position == 1 and low[i] < L4_aligned[i-1]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > H4_aligned[i-1]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals