#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Camarilla pivot breakout with volume confirmation and 1d chop regime filter
# - Long when price breaks above H3 Camarilla level (4h) AND volume > 1.5x 20-period average AND 1d chop > 61.8 (range)
# - Short when price breaks below L3 Camarilla level (4h) AND volume > 1.5x 20-period average AND 1d chop > 61.8 (range)
# - Exit when price reverts to Pivot Point (PP) level (4h)
# - Uses discrete position sizing 0.20 to limit fee churn
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla pivots work well in ranging markets where mean reversion at extremes is effective
# - Volume confirmation ensures breakouts have conviction
# - Chop filter ensures we only trade in ranging conditions where mean reversion works
# - 1h timeframe for precise entry timing, 4h for signal direction, 1d for regime filter

name = "1h_4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 4h Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 4h
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_4h = (high_4h + low_4h + close_4h) / 3.0
    # Range = High - Low
    range_4h = high_4h - low_4h
    
    # Resistance levels
    r4_4h = pp_4h + range_4h * 1.1 / 2  # H4
    r3_4h = pp_4h + range_4h * 1.1 / 4  # H3
    r2_4h = pp_4h + range_4h * 1.1 / 6  # H2
    r1_4h = pp_4h + range_4h * 1.1 / 12 # H1
    
    # Support levels
    s1_4h = pp_4h - range_4h * 1.1 / 12 # L1
    s2_4h = pp_4h - range_4h * 1.1 / 6  # L2
    s3_4h = pp_4h - range_4h * 1.1 / 4  # L3
    s4_4h = pp_4h - range_4h * 1.1 / 2  # L4
    
    # Pre-compute 1d chop regime (choppiness index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # ATR(14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Chop = 100 * log10(tr_sum / range_max_min) / log10(14)
    chop = 100 * np.log10(tr_sum / range_max_min) / np.log10(14)
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])  # align indices
    
    # Chop regime: > 61.8 = ranging (good for mean reversion at extremes)
    chop_range = chop > 61.8
    
    # Align HTF indicators to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(pp_4h_aligned[i]) or np.isnan(chop_range_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 (r3_4h) AND volume spike AND chop range
            if (close[i] > r3_4h_aligned[i] and 
                volume_spike[i] and 
                chop_range_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below L3 (s3_4h) AND volume spike AND chop range
            elif (close[i] < s3_4h_aligned[i] and 
                  volume_spike[i] and 
                  chop_range_aligned[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price reverts to Pivot Point (PP) level
            exit_long = (position == 1 and close[i] <= pp_4h_aligned[i])
            exit_short = (position == -1 and close[i] >= pp_4h_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals