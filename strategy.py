#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 1d Camarilla R3 + volume > 2x 20-period avg + CHOP(12h) < 38.2 (trending)
# Short when price breaks below 1d Camarilla S3 + volume > 2x 20-period avg + CHOP(12h) < 38.2
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-30 trades/year.
# Camarilla levels provide precise intraday support/resistance. Volume spike confirms institutional interest.
# CHOP filter ensures we only trade in trending markets, avoiding chop/range bound conditions.
# Works in bull markets (breakouts continue) and bear markets (breakdowns accelerate) by requiring trending regime.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h Indicator: Choppiness Index (CHOP) - regime filter ===
    chop_window = 14
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate sum of TR over window
    tr_sum = pd.Series(tr).rolling(window=chop_window, min_periods=chop_window).sum().values
    # Calculate highest high and lowest low over window
    hh = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    ll = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    
    # Avoid division by zero and invalid values
    denominator = hh - ll
    chop = np.where(denominator != 0, 100 * np.log10(tr_sum / denominator) / np.log10(chop_window), 50.0)
    # Handle edge cases
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50.0, chop)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(chop_window, 20) + 5  # CHOP(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # CHOP filter: trending market (CHOP < 38.2)
        chop_filter = chop[i] < 38.2
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3
        # 2. Volume confirmation
        # 3. Trending regime (CHOP < 38.2)
        if (close[i] > r3_aligned[i]) and \
           vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3
        # 2. Volume confirmation
        # 3. Trending regime (CHOP < 38.2)
        elif (close[i] < s3_aligned[i]) and \
             vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVol2x_CHOP_Filter_v2"
timeframe = "12h"
leverage = 1.0