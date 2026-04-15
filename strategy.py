#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# Long when price breaks above Camarilla R3 (1d) + volume > 2x 24-period avg + CHOP(14) < 38.2 (trending)
# Short when price breaks below Camarilla S3 (1d) + volume > 2x 24-period avg + CHOP(14) < 38.2
# Uses discrete position sizing (0.30) to balance profit and drawdown. Designed for 12-30 trades/year.
# Camarilla levels provide intraday support/resistance. Volume spike confirms breakout strength.
# Chop filter ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
# Works in bull markets (buying strength) and bear markets (selling weakness) by requiring trending regime.

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
    
    # === 1d Indicator: Camarilla Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels
    rang = high_1d - low_1d
    camarilla_r3 = close_1d + (rang * 1.1 / 4)
    camarilla_s3 = close_1d - (rang * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 12h Indicator: Choppiness Index (CHOP) ===
    # CHOP = 100 * log10(sum(TR over period) / (max(high) - min(low))) / log10(period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    chop_period = 14
    atr_sum = np.zeros_like(tr)
    max_high = np.zeros_like(high)
    min_low = np.zeros_like(low)
    
    for i in range(chop_period, n):
        atr_sum[i] = np.sum(tr[i-chop_period+1:i+1])
        max_high[i] = np.max(high[i-chop_period+1:i+1])
        min_low[i] = np.min(low[i-chop_period+1:i+1])
    
    chop = np.full(n, 50.0)  # default neutral
    valid = (max_high - min_low) > 0
    chop[valid] = 100 * np.log10(atr_sum[valid] / (max_high[valid] - min_low[valid])) / np.log10(chop_period)
    
    # === Volume Confirmation ===
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, chop_period, 24) + 5
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 24-period volume SMA
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # Chop filter: trending market (CHOP < 38.2)
        chop_filter = chop[i] < 38.2
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_sma_24[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (1d)
        # 2. Volume confirmation
        # 3. Trending regime (CHOP < 38.2)
        if (close[i] > camarilla_r3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (1d)
        # 2. Volume confirmation
        # 3. Trending regime (CHOP < 38.2)
        elif (close[i] < camarilla_s3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVolSpike_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0