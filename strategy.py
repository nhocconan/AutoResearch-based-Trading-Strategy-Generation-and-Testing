#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action relative to weekly pivot levels with volume confirmation
# Go long when price crosses above weekly R3 with volume > 1.5x 6h average volume
# Go short when price crosses below weekly S3 with volume > 1.5x 6h average volume
# Exit when price returns to weekly pivot (PP) or reverses at opposite extreme
# Pivot levels calculated from previous week: PP = (H+L+C)/3, R3 = H + 2*(PP-L), S3 = L - 2*(H-PP)
# Uses weekly pivot for directional bias and 6s for entry timing with volume filter
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_weekly_pivot_r3s3_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP, R3, S3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point: PP = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # R3 = High + 2*(PP - Low)
    r3 = weekly_high + 2.0 * (pp - weekly_low)
    # S3 = Low - 2*(High - PP)
    s3 = weekly_low - 2.0 * (weekly_high - pp)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 week for look-ahead prevention)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # 6h volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to weekly pivot or reverses at S3
            elif close[i] <= pp_aligned[i] or close[i] <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to weekly pivot or reverses at R3
            elif close[i] >= pp_aligned[i] or close[i] >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price crosses weekly R3/S3 with volume confirmation
            # Bullish breakout: price crosses above R3
            bullish_break = close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1]
            # Bearish breakdown: price crosses below S3
            bearish_break = close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1]
            
            # Long: bullish breakout above R3 with volume spike
            if (bullish_break and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish breakdown below S3 with volume spike
            elif (bearish_break and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals