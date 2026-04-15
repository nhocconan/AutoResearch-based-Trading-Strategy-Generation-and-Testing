#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 1d Camarilla R3 + volume > 2x 24-period avg + CHOP(14) < 38.2 (trending)
# Short when price breaks below 1d Camarilla S3 + volume > 2x 24-period avg + CHOP(14) < 38.2 (trending)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 12h timeframe.
# Camarilla levels provide intraday support/resistance. Volume confirms breakout strength.
# CHOP filter ensures we only trade in trending markets, avoiding chop/range bound conditions.
# Works in bull markets (trend continuation) and bear markets (strong downtrends) by requiring trending regime.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Camarilla levels
    range_1d = high_1d - low_1d
    r3 = pp + (range_1d * 1.1 / 4.0)  # R3 level
    s3 = pp - (range_1d * 1.1 / 4.0)  # S3 level
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h Indicator: Choppiness Index (CHOP) for regime filter ===
    high_12h = high
    low_12h = low
    close_12h = close
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]
    tr2[0] = np.abs(high_12h[0] - close_12h[0])
    tr3[0] = np.abs(low_12h[0] - close_12h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate CHOP(14)
    chop_period = 14
    atr_sum = np.zeros_like(tr)
    atr_sum[chop_period-1] = np.sum(tr[:chop_period])
    for i in range(chop_period, len(tr)):
        atr_sum[i] = atr_sum[i-1] - tr[i-chop_period] + tr[i]
    
    # Calculate highest high and lowest low over chop_period
    hh = np.zeros_like(high_12h)
    ll = np.zeros_like(low_12h)
    for i in range(chop_period-1, len(high_12h)):
        hh[i] = np.max(high_12h[i-chop_period+1:i+1])
        ll[i] = np.min(low_12h[i-chop_period+1:i+1])
    
    # Avoid division by zero
    chop = np.where((hh - ll) != 0, 
                    100 * np.log10(atr_sum / np.sqrt(chop_period) / (hh - ll)) / np.log10(chop_period), 
                    50.0)  # default to neutral when range is zero
    
    # === 12h Indicator: Volume SMA for confirmation ===
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(24, chop_period) + 5  # volume(24) + CHOP(14) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 24-period volume SMA
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3
        # 2. Volume confirmation
        # 3. Trending regime (CHOP < 38.2)
        if (close[i] > r3_aligned[i]) and \
           vol_confirm and (chop[i] < 38.2):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3
        # 2. Volume confirmation
        # 3. Trending regime (CHOP < 38.2)
        elif (close[i] < s3_aligned[i]) and \
             vol_confirm and (chop[i] < 38.2):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVol2x_CHOP_Filter_v3"
timeframe = "12h"
leverage = 1.0