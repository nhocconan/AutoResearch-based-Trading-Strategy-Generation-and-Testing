#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversal with 12h EMA(50) filter and volume confirmation
# Captures mean reversion at extreme pivot levels (R4/S4) aligned with higher timeframe trend.
# Uses 12h EMA for trend filter to ensure trades are in direction of higher timeframe momentum.
# Volume confirmation filters for institutional participation at reversal points.
# Designed for low frequency: target 12-37 trades/year to minimize fee drag on 6h timeframe.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).

name = "6h_camarilla_pivot_12h_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot points and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_12h = close_12h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    close_12h_series = pd.Series(df_12h['close'].values)
    high_12h_series = pd.Series(df_12h['high'].values)
    low_12h_series = pd.Series(df_12h['low'].values)
    
    # Use previous bar's data to avoid look-ahead
    prev_close = close_12h_series.shift(1).values
    prev_high = high_12h_series.shift(1).values
    prev_low = low_12h_series.shift(1).values
    
    # Calculate pivot levels
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align pivot levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Reversal conditions at extreme levels
        # Long when price touches/slightly exceeds S4 and reverses up
        long_setup = (close[i] <= s4_aligned[i] * 1.001 and  # Allow small overshoot
                      close[i] > s3_aligned[i] and           # Above S3
                      close[i] > close[i-1])                 # Price rising from previous bar
        
        # Short when price touches/slightly exceeds R4 and reverses down
        short_setup = (close[i] >= r4_aligned[i] * 0.999 and # Allow small undershoot
                       close[i] < r3_aligned[i] and          # Below R3
                       close[i] < close[i-1])                # Price falling from previous bar
        
        # Trend filter: 12h EMA direction
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions: reverse at opposite extreme levels
        exit_long = close[i] >= r3_aligned[i]  # Exit at R3
        exit_short = close[i] <= s3_aligned[i] # Exit at S3
        
        if position == 1:  # Long position
            # Exit at R3 or trend reversal
            if exit_long or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit at S3 or trend reversal
            if exit_short or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: S4 reversal + uptrend + volume confirmation
            if long_setup and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: R4 reversal + downtrend + volume confirmation
            elif short_setup and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals