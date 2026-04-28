#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme readings with volume confirmation and 12h EMA50 trend filter.
# Enter long when Williams %R < -80 (oversold) with volume spike and price above 12h EMA50.
# Enter short when Williams %R > -20 (overbought) with volume spike and price below 12h EMA50.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-100 total trades over 4 years.
# Williams %R is a proven mean-reversion oscillator that works in both bull and bear markets.
# Combined with volume confirmation and trend filter to avoid false signals in strong trends.

name = "4h_WilliamsR_Extreme_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:  # Need at least 14 days for Williams %R
        return np.zeros(n)
    
    # Calculate Williams %R on 1d data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    williams_r = np.full(n_1d, np.nan)
    
    # Williams %R calculation with 14-day period
    for i in range(13, n_1d):  # Start from index 13 to have 14 periods (0-13)
        highest_high = np.max(high_1d[i-13:i+1])  # Highest high of last 14 periods
        lowest_low = np.min(low_1d[i-13:i+1])     # Lowest low of last 14 periods
        if highest_high != lowest_low:  # Avoid division by zero
            williams_r[i] = ((highest_high - close_1d[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # Neutral value when range is zero
    
    # Forward fill to get most recent Williams %R values
    williams_r = pd.Series(williams_r).ffill().values
    
    # Align 1d Williams %R to 4h timeframe with 1-bar delay for confirmation
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA50
        above_ema = close[i] > ema_50_12h_aligned[i]
        below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Williams %R extreme conditions with volume confirmation
        # Williams %R: -100 to 0, where < -80 is oversold, > -20 is overbought
        long_signal = williams_r_aligned[i] < -80 and volume_spike[i]
        short_signal = williams_r_aligned[i] > -20 and volume_spike[i]
        
        # Exit conditions: Williams %R returns to neutral zone or trend reversal
        long_exit = williams_r_aligned[i] > -50 or below_ema  # Exit when not oversold or trend turns bearish
        short_exit = williams_r_aligned[i] < -50 or above_ema  # Exit when not overbought or trend turns bullish
        
        # Handle entries and exits
        if long_signal and above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and below_ema and position >= 0:
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