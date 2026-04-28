#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Uses 12h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years).
# Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period), Lips (5-period).
# Long when Lips > Teeth > Jaw (bullish alignment), short when Lips < Teeth < Jaw (bearish).
# 1d EMA50 provides trend filter: long only when close > EMA50, short only when close < EMA50.
# Volume spike (>1.5x 20-bar average) confirms breakout strength.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.
# Works in both bull and bear markets via trend filter + Alligator alignment logic.

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMMA of median price
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMMA of median price
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA50 direction (price above/below EMA50)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Williams Alligator conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Entry conditions with volume confirmation
        long_entry = bullish_alignment and price_above_ema and volume_spike[i]
        short_entry = bearish_alignment and price_below_ema and volume_spike[i]
        
        # Exit conditions: opposite Alligator alignment or trend reversal
        long_exit = not bullish_alignment or close[i] < ema_50_1d_aligned[i]
        short_exit = not bearish_alignment or close[i] > ema_50_1d_aligned[i]
        
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