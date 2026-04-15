# 12h_DailyPivot_Volume_ADX_Filter
# Hypothesis: On 12h timeframe, use daily Camarilla pivot levels with volume confirmation and ADX filter.
# Long at support (S1/S2) in low volatility (ADX < 25) with volume spike.
# Short at resistance (R1/R2) in low volatility with volume spike.
# Works in bull/bear: mean reversion at pivots during low volatility, avoids trending markets.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    # R4 = close + 1.5 * (high - low) etc., but we use standard Camarilla
    camarilla_high = (high_1d - low_1d) * 1.1 / 12
    camarilla_low = (high_1d - low_1d) * 1.1 / 12
    
    # Pivot levels for previous day (using yesterday's data)
    # R4 = close + 1.5*(high-low), R3 = close + 1.0*(high-low), etc.
    # We'll use R3, R2, S2, S3 for entries
    range_1d = high_1d - low_1d
    r3 = close_1d + 1.0 * range_1d
    r2 = close_1d + 0.5 * range_1d
    s2 = close_1d - 0.5 * range_1d
    s3 = close_1d - 1.0 * range_1d
    
    # Align to 12h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # ADX calculation for regime filter (avoid trending markets)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current > 1.5x median of last 24 bars (2 days)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long at support (S2/S3) in low volatility with volume spike
        long_condition = (adx[i] < 25) and (
            (close[i] <= s2_aligned[i] * 1.001) or  # Allow small buffer
            (close[i] <= s3_aligned[i] * 1.001)
        ) and (volume[i] > vol_threshold[i])
        
        # Short at resistance (R2/R3) in low volatility with volume spike
        short_condition = (adx[i] < 25) and (
            (close[i] >= r2_aligned[i] * 0.999) or  # Allow small buffer
            (close[i] >= r3_aligned[i] * 0.999)
        ) and (volume[i] > vol_threshold[i])
        
        # Exit when volatility increases (trending market) or price moves to opposite level
        exit_long = (i > 0 and signals[i-1] > 0 and (
            adx[i] >= 25 or  # Trending market
            close[i] >= r2_aligned[i] * 0.999 or  # Move to resistance
            close[i] <= s3_aligned[i] * 1.001  # Move to strong support (stop)
        ))
        
        exit_short = (i > 0 and signals[i-1] < 0 and (
            adx[i] >= 25 or  # Trending market
            close[i] <= s2_aligned[i] * 1.001 or  # Move to support
            close[i] >= r3_aligned[i] * 0.999  # Move to strong resistance (stop)
        ))
        
        if long_condition:
            signals[i] = 0.25
        elif short_condition:
            signals[i] = -0.25
        elif exit_long or exit_short:
            signals[i] = 0.0
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_DailyPivot_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0