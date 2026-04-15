#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# Long when price breaks above Camarilla R3 (1d) + volume > 2x 20-period avg + CHOP(14) < 61.8 (trending regime)
# Short when price breaks below Camarilla S3 (1d) + volume > 2x 20-period avg + CHOP(14) < 61.8
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (20-40/year).
# Camarilla levels provide intraday support/resistance. Volume spike confirms breakout strength.
# Chop filter ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.

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
    
    # Calculate pivot point and ranges
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r3 = pp + (range_hl * 1.1 / 4.0)  # R3 = PP + (H-L)*1.1/4
    s3 = pp - (range_hl * 1.1 / 4.0)  # S3 = PP - (H-L)*1.1/4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 4h Indicators: Volume Spike and Choppiness Index ===
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) - values < 38.2 = trending, > 61.8 = ranging
    chop_window = 14
    atr_chop = np.zeros(n)
    for i in range(chop_window, n):
        atr_chop[i] = np.sum(np.maximum(np.abs(high[i-chop_window+1:i+1] - low[i-chop_window+1:i+1]), 
                                        np.abs(np.roll(close[i-chop_window+1:i+1], 1)[1:] - low[i-chop_window+1:i+1]))) if i >= chop_window else 0
    
    # Simplified ATR calculation for chop
    atr_14 = np.zeros(n)
    for i in range(1, n):
        atr_14[i] = np.max([
            np.abs(high[i] - low[i]),
            np.abs(high[i] - close[i-1]),
            np.abs(low[i] - close[i-1])
        ])
    
    # Wilder's smoothing for ATR
    atr_smooth = np.zeros_like(atr_14)
    atr_smooth[chop_window-1] = np.mean(atr_14[:chop_window])
    for i in range(chop_window, len(atr_14)):
        atr_smooth[i] = (atr_smooth[i-1] * (chop_window-1) + atr_14[i]) / chop_window
    
    # Calculate Chop: LOG10(SUM(ATR14)/ (HHV(HIGH)-LLV(LOW))) * 100
    chop = np.zeros(n)
    for i in range(chop_window, n):
        hh = np.max(high[i-chop_window+1:i+1])
        ll = np.min(low[i-chop_window+1:i+1])
        sum_atr = np.sum(atr_smooth[i-chop_window+1:i+1])
        if hh > ll and sum_atr > 0:
            chop[i] = np.log10(sum_atr / (hh - ll)) * 100
        else:
            chop[i] = 50.0  # neutral
    
    # Align chop to ensure proper timing (though calculated on 4h, we align for consistency)
    chop_aligned = chop  # already on 4h timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, chop_window) + 5  # 1d data + volume(20) + chop(14)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA (strong breakout)
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Chop filter: trending regime (CHOP < 61.8)
        chop_filter = chop_aligned[i] < 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (1d)
        # 2. Volume confirmation (strong breakout)
        # 3. Trending regime (CHOP < 61.8)
        if (close[i] > r3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (1d)
        # 2. Volume confirmation (strong breakout)
        # 3. Trending regime (CHOP < 61.8)
        elif (close[i] < s3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R3S3_1dVolSpike_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0