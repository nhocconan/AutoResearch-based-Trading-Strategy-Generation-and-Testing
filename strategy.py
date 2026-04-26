#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_VolumeSpike_ChopFilter_v1
Hypothesis: 1d Camarilla pivot breakout with volume spike and choppiness regime filter.
- Uses 1d timeframe for low trade frequency (target: 30-100 total trades over 4 years)
- Weekly trend filter ensures alignment with higher timeframe direction
- Camarilla R3/S3 levels act as strong support/resistance for breakouts
- Volume spike confirms institutional participation
- Choppiness filter avoids whipsaws in ranging markets
- Designed for 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
- Works in bull/bear markets by combining pivot structure with volume/regime filters
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Previous day's OHLC for Camarilla calculation (using 1d data directly)
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Camarilla levels: R3, S3 (strongest levels)
    # R3 = Close + 1.1*(High-Low)/2
    # S3 = Close - 1.1*(High-Low)/2
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2
    s3 = prev_close - 1.1 * camarilla_range / 2
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Choppiness Index (14-period) to avoid ranging markets
    # CHOP = 100 * log10(sum(ATR) / (max(high)-min(low))) / log10(N)
    tr1 = pd.Series(high).rolling(window=14, min_periods=14).max().values - pd.Series(low).rolling(window=14, min_periods=14).min().values
    tr2 = abs(pd.Series(high).shift(1) - pd.Series(low).shift(1))
    tr3 = abs(pd.Series(high).shift(1) - pd.Series(close).shift(1))
    tr4 = abs(pd.Series(low).shift(1) - pd.Series(close).shift(1))
    true_range = pd.concat([tr2, tr3, tr4], axis=1).max(axis=1).values
    atr_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    chop_filter = chop < 61.8  # Trending market (below 61.8 = trending, above = ranging)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 14 for chop)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Entry conditions
        bullish_breakout = close[i] > r3[i] and volume_spike[i] and chop_filter[i] and weekly_uptrend
        bearish_breakout = close[i] < s3[i] and volume_spike[i] and chop_filter[i] and weekly_downtrend
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike in trending market AND weekly uptrend
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike in trending market AND weekly downtrend
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S3 (reversal) OR loss of momentum
            if close[i] < s3[i] or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R3 (reversal) OR loss of momentum
            if close[i] > r3[i] or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_VolumeSpike_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0