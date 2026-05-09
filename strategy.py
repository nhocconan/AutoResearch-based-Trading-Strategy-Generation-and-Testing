#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Pivot Breakout with Weekly Trend Filter and Volume Confirmation
# Uses weekly R3/S3 breakouts for strong momentum, confirmed by 1-week EMA200 trend filter.
# Volume > 1.5x 20-period EMA ensures institutional participation. Designed to capture
# major trend moves while avoiding chop. Targets 15-25 trades/year for low fee drag.
name = "1d_WeeklyPivot_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points and EMA200 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Previous week's OHLC for Weekly Pivot points (R3, S3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Pivot levels
    # Pivot Point = (High + Low + Close) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # Range = High - Low
    range_1w = high_1w - low_1w
    # R3 = High + 2*(Pivot - Low) = Pivot + Range*1.1666
    # S3 = Low - 2*(High - Pivot) = Pivot - Range*1.1666
    r3 = pp + (range_1w * 1.1666)
    s3 = pp - (range_1w * 1.1666)
    
    # Use previous week's levels (shift by 1 to avoid look-ahead)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    
    # Align to 1d timeframe
    r3_1d = align_htf_to_ltf(prices, df_1w, r3_shifted)
    s3_1d = align_htf_to_ltf(prices, df_1w, s3_shifted)
    
    # Weekly EMA200 trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and above weekly EMA200
            if (price > r3_1d[i] and vol_spike[i] and price > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and below weekly EMA200
            elif (price < s3_1d[i] and vol_spike[i] and price < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S3 (mean reversion to support)
            if price < s3_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R3 (mean reversion to resistance)
            if price > r3_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals