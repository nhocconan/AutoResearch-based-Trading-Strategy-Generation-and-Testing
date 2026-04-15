#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume confirmation and choppiness regime filter
# Long when price breaks above 1d Camarilla R3 + volume > 1.5x 20-period avg + 12h Chop < 61.8 (trending)
# Short when price breaks below 1d Camarilla S3 + volume > 1.5x 20-period avg + 12h Chop < 61.8 (trending)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Camarilla levels provide institutional support/resistance. Chop filter avoids false breakouts in ranging markets.

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
    # Calculate range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pp + (range_1d * 1.1 / 4.0)  # R3 = PP + (High - Low) * 1.1/4
    s3 = pp - (range_1d * 1.1 / 4.0)  # S3 = PP - (High - Low) * 1.1/4
    
    # Align to 12h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h Indicator: Choppiness Index (CHOP) regime filter ===
    # CHOP < 38.2 = trending, CHOP > 61.8 = ranging (we want trending markets)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]
    tr2[0] = np.abs(high_12h[0] - close_12h[0])
    tr3[0] = np.abs(low_12h[0] - close_12h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period)
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Sum of ATR over CHOP period (14)
    sum_atr_14 = np.zeros_like(atr)
    for i in range(atr_period-1, len(atr)):
        if i == atr_period-1:
            sum_atr_14[i] = np.sum(atr[i-atr_period+1:i+1])
        else:
            sum_atr_14[i] = sum_atr_14[i-1] - atr[i-atr_period] + atr[i]
    
    # Max high and min low over CHOP period (14)
    max_high_14 = np.zeros_like(high_12h)
    min_low_14 = np.zeros_like(low_12h)
    for i in range(len(high_12h)):
        if i < 13:
            max_high_14[i] = np.max(high_12h[:i+1])
            min_low_14[i] = np.min(low_12h[:i+1])
        else:
            max_high_14[i] = np.max(high_12h[i-13:i+1])
            min_low_14[i] = np.min(low_12h[i-13:i+1])
    
    # Choppiness Index: CHOP = 100 * LOG10(sum(ATR14) / (MAXHIGH14 - MINLOW14)) / LOG10(14)
    range_14 = max_high_14 - min_low_14
    # Avoid division by zero
    chop = np.where(range_14 > 0, 100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 50)
    # For very low volatility, set to 50 (neutral)
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20) + 20  # 1d data needs ~30 bars, volume needs 20
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3
        # 2. Volume confirmation
        # 3. Trending market (CHOP < 61.8)
        if (close[i] > r3_aligned[i]) and \
           vol_confirm and \
           (chop[i] < 61.8):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3
        # 2. Volume confirmation
        # 3. Trending market (CHOP < 61.8)
        elif (close[i] < s3_aligned[i]) and \
             vol_confirm and \
             (chop[i] < 61.8):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVolume_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0