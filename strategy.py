#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal + 1d trend filter + volume confirmation.
# Long at S3 support during bullish daily trend with volume > 1.5x 20-period average.
# Short at R3 resistance during bearish daily trend with volume confirmation.
# Uses Camarilla levels derived from prior 1d session (high, low, close).
# Target: 60-120 total trades over 4 years (15-30/year) to stay within optimal range.

name = "6h_camarilla_reversal_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels (based on prior day)
    df_1d = get_htf_data(prices, '1d')
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for current day using prior day's HLC
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    rng = d_high - d_low
    r3 = d_close + rng * 1.1 / 4
    s3 = d_close - rng * 1.1 / 4
    
    # Daily trend filter: bullish/bearish day based on close vs open
    d_open = df_1d['open'].values
    daily_bullish = d_close > d_open
    daily_bearish = d_close < d_open
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]) or \
           np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price rises to S1 or daily turn bearish
            s1 = d_close[i] - (d_high[i] - d_low[i]) * 1.1 / 12
            s1_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(d_close, s1))[i] if not np.isnan(s1) else np.nan
            if (low[i] <= s3_aligned[i] and high[i] >= s1_aligned) or daily_bearish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price falls to R1 or daily turn bullish
            r1 = d_close[i] + (d_high[i] - d_low[i]) * 1.1 / 12
            r1_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(d_close, r1))[i] if not np.isnan(r1) else np.nan
            if (high[i] >= r3_aligned[i] and low[i] <= r1_aligned) or daily_bullish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for reversals at S3/R3 with volume confirmation and daily trend filter
            if volume_filter:
                # Long: reversal from S3 support during bullish day
                if low[i] <= s3_aligned[i] and daily_bullish_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: rejection at R3 resistance during bearish day
                elif high[i] >= r3_aligned[i] and daily_bearish_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals