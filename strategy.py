#!/usr/bin/env python3
# 1d_weekly_higher_high_lower_low_volume_v1
# Hypothesis: Daily strategy using weekly higher highs/lower lows with volume confirmation.
# Long: Price makes a weekly higher high (close > weekly high_1w) with volume > 2.0x 20-day average.
# Short: Price makes a weekly lower low (close < weekly low_1w) with volume > 2.0x 20-day average.
# Exit: Price crosses weekly EMA(21) in opposite direction.
# Uses discrete position sizing (0.25) to limit fee drag. Works in bull markets via breakouts
# and bear markets via mean reversion from extremes. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_higher_high_lower_low_volume_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for higher high/lower low and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA(21) for exit
    close_1w_s = pd.Series(close_1w)
    ema_21_1w = close_1w_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Weekly higher high: current week high > previous week high
    # Weekly lower low: current week low < previous week low
    higher_high = high_1w > np.roll(high_1w, 1)
    lower_low = low_1w < np.roll(low_1w, 1)
    # Set first value to False (no previous week)
    higher_high[0] = False
    lower_low[0] = False
    
    # Align weekly signals to daily
    higher_high_aligned = align_htf_to_ltf(prices, df_1w, higher_high.astype(float))
    lower_low_aligned = align_htf_to_ltf(prices, df_1w, lower_low.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(higher_high_aligned[i]) or np.isnan(lower_low_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-day average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below weekly EMA(21)
            if close[i] < ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above weekly EMA(21)
            if close[i] > ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Weekly higher high with volume confirmation
            if (higher_high_aligned[i] > 0.5 and    # Weekly higher high
                volume_confirmed):                  # Volume spike
                position = 1
                signals[i] = 0.25
            # Short entry: Weekly lower low with volume confirmation
            elif (lower_low_aligned[i] > 0.5 and    # Weekly lower low
                  volume_confirmed):                # Volume spike
                position = -1
                signals[i] = -0.25
    
    return signals