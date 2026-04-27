#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1-day EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R1 level, short when breaks below S1 level.
# Trend filter: price must be above/below 1-day EMA34 to align with higher timeframe direction.
# Volume confirmation (>2x 20-period average) ensures institutional participation and filters noise.
# Exit when price returns to the Pivot Point (PP) level or reverses to opposite S1/R1 level.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Camarilla levels provide precise intraday support/resistance, effective in both trending and ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Previous day's Camarilla levels (using prior day's H/L/C)
    # Calculate for each 1d bar, then align to 4h
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First day will have invalid values (rolled), but alignment will handle timing
    
    # Camarilla calculations for previous day
    R1 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    S1 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
    PP = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    
    # Align Camarilla levels to 4h timeframe (they are constant throughout the 4h bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # Volume filter: volume > 2x 20-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(PP_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R1, above 1d EMA34, volume spike
        if (close[i] > R1_aligned[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below S1, below 1d EMA34, volume spike
        elif (close[i] < S1_aligned[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to PP level or reverses to opposite level
        elif position == 1 and (close[i] <= PP_aligned[i] or close[i] < S1_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] >= PP_aligned[i] or close[i] > R1_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0