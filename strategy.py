#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above Camarilla R3 (1d) + volume > 2x 20-period avg + chop < 61.8 (trending)
# Short when price breaks below Camarilla S3 (1d) + volume > 2x 20-period avg + chop < 61.8
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Camarilla levels provide intraday support/resistance. Chop filter avoids ranging markets.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by requiring trending regime.

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
    # Calculate range
    rng = high_1d - low_1d
    # Camarilla levels
    r3 = pp + (rng * 1.1 / 4.0)  # R3 = PP + 1.1 * (High - Low) / 4
    s3 = pp - (rng * 1.1 / 4.0)  # S3 = PP - 1.1 * (High - Low) / 4
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h Indicator: Choppiness Index (CHOP) ===
    # CHOP = 100 * log10(sum(ATR, n) / (max(high,n) - min(low,n))) / log10(n)
    chop_window = 14
    # Calculate True Range
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - close_shift)
    tr3 = np.abs(low - close_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of ATR over window
    atr_sum = np.zeros_like(tr)
    for i in range(chop_window, len(tr)):
        atr_sum[i] = np.sum(tr[i-chop_window+1:i+1])
    # For warmup period, use cumulative sum
    if chop_window > 0:
        atr_sum[chop_window-1] = np.sum(tr[:chop_window])
    
    # Max high and min low over window
    max_high = np.zeros_like(high)
    min_low = np.zeros_like(low)
    for i in range(chop_window, len(high)):
        max_high[i] = np.max(high[i-chop_window+1:i+1])
        min_low[i] = np.min(low[i-chop_window+1:i+1])
    # For warmup period
    if chop_window > 0:
        max_high[chop_window-1] = np.max(high[:chop_window])
        min_low[chop_window-1] = np.min(low[:chop_window])
    
    # Avoid division by zero
    denominator = max_high - min_low
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    chop = np.zeros_like(close)
    for i in range(chop_window, len(close)):
        if atr_sum[i] > 0 and denominator[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / denominator[i]) / np.log10(chop_window)
        else:
            chop[i] = 50.0  # neutral value
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, chop_window) + 20  # 1d data needs ~30 bars + chop + volume
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Chop filter: trending market (CHOP < 61.8)
        chop_filter = chop[i] < 61.8
        
        # Skip if any required data is NaN or invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3
        # 2. Volume confirmation
        # 3. Trending regime (CHOP < 61.8)
        if (close[i] > r3_aligned[i]) and \
           vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3
        # 2. Volume confirmation
        # 3. Trending regime (CHOP < 61.8)
        elif (close[i] < s3_aligned[i]) and \
             vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVol2x_CHOP_Filter_v1"
timeframe = "12h"
leverage = 1.0