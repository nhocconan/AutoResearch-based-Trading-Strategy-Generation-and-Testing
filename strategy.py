# [Experiment #103882] 12h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Confirmation
# Hypothesis: Camarilla R3/S3 levels on 1d timeframe act as strong support/resistance. 
# Breakouts with 1d EMA34 trend alignment and volume confirmation reduce false signals.
# Works in bull markets (breakouts with trend) and bear (mean-reversion at extremes).
# Target: 12-37 trades/year on 12h timeframe to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for EMA34
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard Camarilla calculation
    range_1d = high_1d - low_1d
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # First day uses its own close
    
    # Camarilla levels
    r3 = close_prev + 1.1 * range_1d / 2
    s3 = close_prev - 1.1 * range_1d / 2
    r4 = close_prev + 1.1 * range_1d
    s4 = close_prev - 1.1 * range_1d
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.8x 20-period average to avoid noise
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        # Entry conditions
        # Long: break above R3 with upward trend and volume
        long_breakout = close[i] > r3_aligned[i]
        long_entry = long_breakout and trend_up and volume_filter[i]
        
        # Short: break below S3 with downward trend and volume
        short_breakout = close[i] < s3_aligned[i]
        short_entry = short_breakout and trend_down and volume_filter[i]
        
        # Exit conditions: opposite S3/R3 levels (mean reversion)
        long_exit = close[i] < s3_aligned[i] and position == 1
        short_exit = close[i] > r3_aligned[i] and position == -1
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
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

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0