#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter.
# Enter long when price breaks above Camarilla R3 and 1d volume > 2x 20-bar average and chop > 61.8 (range).
# Enter short when price breaks below Camarilla S3 and 1d volume > 2x 20-bar average and chop > 61.8.
# Exit when price crosses Camarilla H4/L4 or chop < 38.2 (trend).
# Uses discrete position sizing (0.25) to minimize fee drag.
# Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag.
# Camarilla levels provide intraday support/resistance, chop filter avoids whipsaw in trends,
# volume confirmation ensures conviction. Works in ranging markets typical of 2025+.

name = "12h_Camarilla_R3S3_Breakout_1dVolumeChop_v1"
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
    
    # Get 1d data for volume and chop filters (MTF structure)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume
    vol_1d = df_1d['volume'].values
    vol_1d_series = pd.Series(vol_1d)
    vol_ma_20_1d = vol_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d chopiness index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # True Range ATR (using TR directly for chop)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max/Min close over 14 periods
    max_close_14 = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_close_14 = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: log10(sumTR14 / (maxClose-minClose)) * 100 / log10(14)
    range_14 = max_close_14 - min_close_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop_1d = np.log10(tr_sum_14 / range_14) * 100 / np.log10(14)
    
    # Align 1d volume and chop to 12h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Camarilla levels from previous 1d (using typical calculation)
    # Camarilla levels are based on previous day's range
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # First day handling
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Calculate Camarilla levels for 1d
    range_1d = prev_high - prev_low
    camarilla_h5 = prev_close + range_1d * 1.1 / 2
    camarilla_h4 = prev_close + range_1d * 1.1 / 4
    camarilla_h3 = prev_close + range_1d * 1.1 / 6
    camarilla_l3 = prev_close - range_1d * 1.1 / 6
    camarilla_l4 = prev_close - range_1d * 1.1 / 4
    camarilla_l5 = prev_close - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2x 20-bar 1d average volume
        # Note: Using current bar's volume vs 1d average - this is a proxy for volume spike
        vol_confirm = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        # Chop regime filter: chop > 61.8 = ranging (good for mean reversion at extremes)
        chop_range = chop_aligned[i] > 61.8
        chop_trend = chop_aligned[i] < 38.2  # Exit condition
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # Break above R3
        breakout_down = low[i] < camarilla_l3_aligned[i-1]  # Break below S3
        
        # Exit conditions
        exit_long = close[i] < camarilla_h4_aligned[i] or chop_trend
        exit_short = close[i] > camarilla_l4_aligned[i] or chop_trend
        
        # Handle entries and exits
        if breakout_up and vol_confirm and chop_range and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and vol_confirm and chop_range and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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