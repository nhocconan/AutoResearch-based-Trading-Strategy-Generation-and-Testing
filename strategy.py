#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Daily chart strategy using 1-week HTF pivot levels with volume confirmation
# Target: Capture trend continuation at weekly support/resistance with volume validation
# Works in bull/bear: Weekly pivots act as dynamic support/resistance; volume filters false breakouts
name = "1d_1w_Pivot_R1S1_Breakout_Volume_Confirmation_v1"
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
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels from previous week (non-lookahead)
    prev_close_w = np.roll(close_1w, 1)
    prev_close_w[0] = np.nan
    prev_high_w = np.roll(high_1w, 1)
    prev_high_w[0] = np.nan
    prev_low_w = np.roll(low_1w, 1)
    prev_low_w[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_w = prev_close_w + (prev_high_w - prev_low_w) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_w = prev_close_w - (prev_high_w - prev_low_w) * 1.1 / 12.0
    
    # Align to daily timeframe (waits for weekly bar close)
    pivot_w_daily = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_daily = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_daily = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_w_daily[i]) or np.isnan(r1_w_daily[i]) or np.isnan(s1_w_daily[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume confirmation
            if price > r1_w_daily[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume confirmation
            elif price < s1_w_daily[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below weekly pivot (mean reversion)
            if price < pivot_w_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above weekly pivot (mean reversion)
            if price > pivot_w_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals