#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams %R extremes with 1d EMA50 trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels: long when %R < -80 (oversold), short when %R > -20 (overbought).
# Enter long when Williams %R crosses above -80 and close > 1d EMA50 and volume > 2x 20-bar average.
# Enter short when Williams %R crosses below -20 and close < 1d EMA50 and volume > 2x 20-bar average.
# Exit when Williams %R crosses above -50 for long or below -50 for short (mean reversion to midline).
# Uses discrete position sizing (0.30) to balance risk and return.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Williams %R is effective in ranging markets and captures reversals in both bull and bear regimes.
# 1d EMA50 filter ensures alignment with higher timeframe trend, reducing whipsaws.
# High volume threshold (2x average) adds confirmation to reduce false signals.

name = "1d_WilliamsR_Extremes_1dEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams %R (HTF structure/reversal signals)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w Williams %R (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1w) / (highest_high - lowest_low)
    # Replace division by zero or near-zero with -100 (fully oversold)
    williams_r = np.where((highest_high - lowest_low) == 0, -100, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w Williams %R to 1d timeframe (with extra delay for indicator confirmation)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r, additional_delay_bars=1)
    
    # Calculate volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d EMA50 bias
        bullish_bias = close[i] > ema_50_1d[i]
        bearish_bias = close[i] < ema_50_1d[i]
        
        # Williams %R conditions
        williams_r_current = williams_r_aligned[i]
        williams_r_previous = williams_r_aligned[i-1]
        
        # Entry conditions: %R crossing extreme levels with confirmation
        long_entry = (williams_r_previous <= -80) and (williams_r_current > -80) and bullish_bias and vol_confirm
        short_entry = (williams_r_previous >= -20) and (williams_r_current < -20) and bearish_bias and vol_confirm
        
        # Exit conditions: %R crossing midline (-50) indicating mean reversion
        long_exit = (williams_r_previous < -50) and (williams_r_current >= -50)
        short_exit = (williams_r_previous > -50) and (williams_r_current <= -50)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals