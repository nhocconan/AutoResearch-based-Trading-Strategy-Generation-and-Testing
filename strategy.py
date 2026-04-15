#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
# Long when: price > Alligator Jaw (13-period SMMA) + price > Alligator Teeth (8-period SMMA) + 
#            Alligator Lips (5-period SMMA) > Alligator Teeth > Alligator Jaw (bullish alignment) +
#            volume > 2.0x 20-period volume SMA + Choppiness Index(14) < 38.2 (trending regime)
# Short when: price < Alligator Jaw + price < Alligator Teeth + 
#             Alligator Lips < Alligator Teeth < Alligator Jaw (bearish alignment) +
#             volume > 2.0x 20-period volume SMA + Choppiness Index(14) < 38.2
# Uses Williams Alligator for trend detection with smoothed moving averages.
# Volume spike confirms momentum. Chop filter ensures we only trade in trending markets.
# Designed for low trade frequency (15-30/year) on 12h timeframe to minimize fee drag.
# Works in bull markets (bullish Alligator alignment) and bear markets (bearish alignment).

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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 12h Indicator: Williams Alligator (SMMA-based) ===
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    
    jaw = smma(typical_price_12h, 13)  # Jaw (13-period)
    teeth = smma(typical_price_12h, 8)  # Teeth (8-period)
    lips = smma(typical_price_12h, 5)   # Lips (5-period)
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # === 1d Indicator: Choppiness Index (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period)
    atr_period = 14
    atr = np.zeros_like(tr)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Highest high and lowest low over 14 periods
    hh = np.full_like(high_1d, np.nan, dtype=float)
    ll = np.full_like(low_1d, np.nan, dtype=float)
    for i in range(atr_period-1, len(high_1d)):
        hh[i] = np.max(high_1d[i-atr_period+1:i+1])
        ll[i] = np.min(low_1d[i-atr_period+1:i+1])
    
    # Choppiness Index = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    chop = np.full_like(close_1d, np.nan, dtype=float)
    sum_tr = np.zeros_like(close_1d)
    if len(tr) >= atr_period:
        # Calculate rolling sum of TR
        sum_tr[atr_period-1] = np.sum(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            sum_tr[i] = sum_tr[i-1] - tr[i-atr_period] + tr[i]
        
        # Calculate Choppiness Index
        denominator = hh - ll
        valid = (denominator > 0) & ~np.isnan(sum_tr) & ~np.isnan(denominator)
        chop[valid] = 100 * np.log10(sum_tr[valid] / denominator[valid]) / np.log10(atr_period)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 20  # Alligator needs 50 bars, volume needs 20
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bullish Alligator alignment: Lips > Teeth > Jaw
        # 2. Price above Jaw (confirmation of uptrend)
        # 3. Volume confirmation
        # 4. Trending regime: Chop < 38.2
        if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) and \
           (close[i] > jaw_aligned[i]) and \
           vol_confirm and \
           (chop_aligned[i] < 38.2):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bearish Alligator alignment: Lips < Teeth < Jaw
        # 2. Price below Jaw (confirmation of downtrend)
        # 3. Volume confirmation
        # 4. Trending regime: Chop < 38.2
        elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) and \
             (close[i] < jaw_aligned[i]) and \
             vol_confirm and \
             (chop_aligned[i] < 38.2):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsAlligator_1dVol2x_CHOP_Filter_v1"
timeframe = "12h"
leverage = 1.0