#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA34 trend filter and volume spike confirmation.
# Uses Camarilla levels from 1d pivots (stronger R4/S4 levels) with 12h EMA34 for trend.
# Long when price breaks above R4 with volume and price > 12h EMA34 (uptrend).
# Short when price breaks below S4 with volume and price < 12h EMA34 (downtrend).
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Designed for 4h timeframe to target 75-200 trades over 4 years (19-50/year).

name = "4h_Camarilla_R4S4_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 1d Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d_prev) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels (R4/S4 are stronger breakout levels than R3/S3)
    R4 = pivot + range_1d * 1.1 / 2.0
    S4 = pivot - range_1d * 1.1 / 2.0
    
    # Align to 4h timeframe (use previous 1d bar's levels)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for EMA34 and pivots
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA34 direction (price above/below EMA34)
        price_above_ema = close[i] > ema_34_12h_aligned[i]
        price_below_ema = close[i] < ema_34_12h_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R4_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S4_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < S4_aligned[i] or close[i] < ema_34_12h_aligned[i]
        short_exit = close[i] > R4_aligned[i] or close[i] > ema_34_12h_aligned[i]
        
        # Handle entries and exits
        if long_breakout and price_above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and price_below_ema and position >= 0:
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