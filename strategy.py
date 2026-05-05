#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Camarilla pivot (R3/S3) breakout with 1w EMA20 trend filter and volume spike confirmation
# Long when price breaks above weekly R3 AND price > 1w EMA20 AND volume > 2.0 * avg_volume(20) on 1d
# Short when price breaks below weekly S3 AND price < 1w EMA20 AND volume > 2.0 * avg_volume(20) on 1d
# Exit when price returns to weekly pivot point (PP) OR volume drops below average
# Uses discrete sizing 0.30 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Weekly Camarilla provides strong support/resistance levels that work in ranging and trending markets
# 1w EMA20 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "1d_Camarilla_R3S3_1wEMA20_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Range = High - Low
    range_1w = high_1w - low_1w
    # Resistance levels: R1 = PP + (Range * 1.1/12), R2 = PP + (Range * 1.1/6), R3 = PP + (Range * 1.1/4)
    # Support levels: S1 = PP - (Range * 1.1/12), S2 = PP - (Range * 1.1/6), S3 = PP - (Range * 1.1/4)
    r3_1w = pp_1w + (range_1w * 1.1 / 4.0)
    s3_1w = pp_1w - (range_1w * 1.1 / 4.0)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Get 1w data ONCE before loop for EMA20 trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3, above 1w EMA20, volume confirmation, in session
            if (close[i] > r3_1w_aligned[i] and close[i-1] <= r3_1w_aligned[i-1] and 
                close[i] > ema20_1w_aligned[i] and volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below weekly S3, below 1w EMA20, volume confirmation, in session
            elif (close[i] < s3_1w_aligned[i] and close[i-1] >= s3_1w_aligned[i-1] and 
                  close[i] < ema20_1w_aligned[i] and volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly pivot point OR volume drops below average
            if close[i] <= pp_1w_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price returns to weekly pivot point OR volume drops below average
            if close[i] >= pp_1w_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals