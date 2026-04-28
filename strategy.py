#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R breakout with 12h EMA50 trend filter and volume confirmation.
# Enter long when Williams %R crosses above -20 from below (oversold breakout) with 12h EMA50 uptrend and volume > 1.8x 20-bar average.
# Enter short when Williams %R crosses below -80 from above (overbought breakdown) with 12h EMA50 downtrend and volume confirmation.
# Exit when Williams %R crosses -50 (mean reversion) or at Camarilla midpoint for symmetry.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Williams %R provides momentum exhaustion signals that work in both bull and bear markets.
# EMA50 on 12h ensures trend alignment. Volume confirmation filters weak breakouts.

name = "4h_WilliamsR_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams %R calculation (14-period)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 14:  # Need sufficient data for Williams %R
        return np.zeros(n)
    
    # Calculate 4h Williams %R (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_4h) / (highest_high - lowest_low) * -100
    
    # Align Williams %R to 4h (shifted by one bar to avoid look-ahead)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 12h EMA (50-period)
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: >1.8x 20-bar average volume (tighter than 2.0x to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 14)  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 12h EMA trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        ema_trend_up = close[i] > ema_50_aligned[i]
        ema_trend_down = close[i] < ema_50_aligned[i]
        
        williams_r_val = williams_r_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -20 (from below), uptrend, volume confirm
            if (williams_r_val > -20 and 
                i > start_idx and williams_r_aligned[i-1] <= -20 and  # crossed above -20
                ema_trend_up and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -80 (from above), downtrend, volume confirm
            elif (williams_r_val < -80 and 
                  i > start_idx and williams_r_aligned[i-1] >= -80 and  # crossed below -80
                  ema_trend_down and vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R crosses below -50 (mean reversion)
            if i > start_idx and williams_r_val < -50 and williams_r_aligned[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R crosses above -50 (mean reversion)
            if i > start_idx and williams_r_val > -50 and williams_r_aligned[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals