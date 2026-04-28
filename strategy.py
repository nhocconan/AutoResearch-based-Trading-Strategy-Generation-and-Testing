#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses 12h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years).
# Camarilla pivot levels (R3, S3) from 1d provide strong support/resistance: long when price breaks above R3, short when breaks below S3.
# 1d EMA34 provides primary trend filter: long only when price > EMA34, short only when price < EMA34.
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.
# Works in both bull and bear markets via trend filter + breakout logic.
# Targets BTC and ETH primarily; SOL is secondary.

name = "12h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Get 12h data for price action and 1d data for Camarilla pivots and EMA34
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    # Pivot = (H + L + C) / 3
    # R3 = H + 2*(Pivot - L)
    # S3 = L - 2*(H - Pivot)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = high_1d + 2.0 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2.0 * (high_1d - pivot_1d)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient history for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction (price above/below EMA34)
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions
        breakout_above_r3 = close[i] > r3_1d_aligned[i]
        breakout_below_s3 = close[i] < s3_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and breakout_above_r3 and vol_confirm
        short_entry = price_below_ema and breakout_below_s3 and vol_confirm
        
        # Exit conditions: price returns to pivot level (mean reversion to fair value)
        long_exit = close[i] < pivot_1d_aligned[i]  # Exit long when price falls below pivot
        short_exit = close[i] > pivot_1d_aligned[i]  # Exit short when price rises above pivot
        
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