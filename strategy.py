#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R3 in 1d uptrend with volume spike (>1.8x 20-period volume MA).
# Short when price breaks below S3 in 1d downtrend with volume spike.
# Exit when price retests the pivot point (PP) or 1d trend reverses.
# Camarilla levels from 1d provide statistically significant support/resistance.
# Breakouts at R3/S3 with volume confirmation indicate institutional participation.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by only trading with the 1d trend.

name = "6h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for price action
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 10:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pp + range_1d * 1.1 / 4.0  # R3 = PP + 1.1 * Range / 4
    s3 = pp - range_1d * 1.1 / 4.0  # S3 = PP - 1.1 * Range / 4
    r4 = pp + range_1d * 1.1 / 2.0  # R4 = PP + 1.1 * Range / 2
    s4 = pp - range_1d * 1.1 / 2.0  # S4 = PP - 1.1 * Range / 2
    
    # Align Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)  # Volume at least 1.8x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for reference
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        pp_val = pp_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: price breaks above R3 AND 1d uptrend AND volume spike
            if close_val > r3_val and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below S3 AND 1d downtrend AND volume spike
            elif close_val < s3_val and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: price retests pivot point (mean reversion)
            if abs(close_val - pp_val) < 0.001 * pp_val:  # Within 0.1% of PP
                exit_signal = True
            # Exit: 1d trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            # Exit: price breaks above R4 (extended move, take profit)
            elif close_val > r4_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: price retests pivot point (mean reversion)
            if abs(close_val - pp_val) < 0.001 * pp_val:  # Within 0.1% of PP
                exit_signal = True
            # Exit: 1d trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            # Exit: price breaks below S4 (extended move, take profit)
            elif close_val < s4_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals