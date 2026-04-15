#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# Long when price breaks above 1d Camarilla R3 level + volume > 2.0x 24-period avg + CHOP(14) > 61.8 (range)
# Short when price breaks below 1d Camarilla S3 level + volume > 2.0x 24-period avg + CHOP(14) > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Camarilla levels provide intraday support/resistance. Chop filter ensures we only trade in ranging markets.
# Works in both bull and bear markets by fading extremes in ranging conditions (mean reversion).

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
    
    # Calculate pivot point
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h Indicator: Choppiness Index (CHOP) ===
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(period)
    # Range: 0-100, >61.8 = ranging, <38.2 = trending
    chop_window = 14
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR (using TR as ATR(1) for CHOP formula)
    atr1 = tr  # ATR(1) is just TR
    # Sum of ATR over period
    sum_atr = np.zeros_like(atr1)
    for i in range(chop_window, len(atr1)):
        sum_atr[i] = np.sum(atr1[i-chop_window+1:i+1])
    # For warmup period, use cumulative sum
    for i in range(chop_window):
        sum_atr[i] = np.sum(atr1[:i+1])
    
    # Calculate max(high) - min(low) over period
    max_high = np.zeros_like(high)
    min_low = np.zeros_like(low)
    for i in range(chop_window, len(high)):
        max_high[i] = np.max(high[i-chop_window+1:i+1])
        min_low[i] = np.min(low[i-chop_window+1:i+1])
    # For warmup period
    for i in range(chop_window):
        max_high[i] = np.max(high[:i+1])
        min_low[i] = np.min(low[:i+1])
    
    range_hl = max_high - min_low
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    # Calculate CHOP
    chop = np.zeros_like(close)
    for i in range(chop_window-1, len(close)):
        if sum_atr[i] > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / range_hl[i]) / np.log10(chop_window)
        else:
            chop[i] = 50.0  # neutral value
    
    # === 12h Indicator: Volume SMA for confirmation ===
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 24)  # 1d data needs 30 bars, volume needs 24
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 24-period volume SMA
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # Chop filter: CHOP > 61.8 (ranging market)
        chop_filter = chop[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_sma_24[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3 level
        # 2. Volume confirmation
        # 3. Ranging market (CHOP > 61.8)
        if (close[i] > r3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3 level
        # 2. Volume confirmation
        # 3. Ranging market (CHOP > 61.8)
        elif (close[i] < s3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVolSpike_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0