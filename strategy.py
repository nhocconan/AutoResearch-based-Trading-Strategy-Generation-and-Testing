#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above Camarilla R3 (1d) + volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Short when price breaks below Camarilla S3 (1d) + volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years.
# Camarilla pivots provide precise intraday support/resistance. Volume spike confirms breakout strength.
# Choppiness filter ensures we only trade in ranging markets where mean reversion at pivot levels works.
# Works in bull markets (buying dips to S3 in range) and bear markets (selling rallies to R3 in range).

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
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 4.0)
    s3 = pivot - (range_1d * 1.1 / 4.0)
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h Indicator: Choppiness Index (CHOP) ===
    # CHOP = 100 * log10(sum(TR over period) / (maxHH - minLL)) / log10(period)
    period_chop = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Rolling sum of TR
    tr_sum = pd.Series(tr).rolling(window=period_chop, min_periods=period_chop).sum().values
    
    # Rolling max/high and min/low
    max_high = pd.Series(high).rolling(window=period_chop, min_periods=period_chop).max().values
    min_low = pd.Series(low).rolling(window=period_chop, min_periods=period_chop).min().values
    
    # Choppiness Index
    chop = np.zeros_like(tr_sum)
    denominator = max_high - min_low
    # Avoid division by zero and log of zero
    valid = (denominator > 0) & (tr_sum > 0)
    chop[valid] = 100 * np.log10(tr_sum[valid] / denominator[valid]) / np.log10(period_chop)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(period_chop, 20) + 5  # CHOP(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (1d)
        # 2. Volume confirmation
        # 3. Choppiness > 61.8 (ranging market)
        if (close[i] > r3_aligned[i]) and vol_confirm and (chop[i] > 61.8):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (1d)
        # 2. Volume confirmation
        # 3. Choppiness > 61.8 (ranging market)
        elif (close[i] < s3_aligned[i]) and vol_confirm and (chop[i] > 61.8):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_CamarillaR3S3_1dVol2x_CHOP_Filter_v1"
timeframe = "12h"
leverage = 1.0