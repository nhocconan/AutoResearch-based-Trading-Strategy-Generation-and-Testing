#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume spike and 1w choppiness regime filter.
# Long when Jaw < Teeth < Lips (bullish alignment) AND volume > 2.0x 20-period 1d average AND weekly CHOP > 61.8 (ranging market).
# Short when Jaw > Teeth > Lips (bearish alignment) AND volume > 2.0x 20-period 1d average AND weekly CHOP > 61.8.
# Exit when Alligator lines cross (Jaw-Teeth or Teeth-Lips crossover) or CHOP < 38.2 (trending regime).
# Uses discrete position size 0.25. Designed to capture mean reversion in ranging markets during both bull and bear regimes.
# Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Williams Alligator (Jaw, Teeth, Lips) ===
    # Jaw: Blue line, 13-period SMMA shifted 8 bars ahead
    # Teeth: Red line, 8-period SMMA shifted 5 bars ahead  
    # Lips: Green line, 5-period SMMA shifted 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Shift: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # First shifted values are invalid
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Bullish alignment: Jaw < Teeth < Lips
    # Bearish alignment: Jaw > Teeth > Lips
    bullish_align = (jaw < teeth) & (teeth < lips)
    bearish_align = (jaw > teeth) & (teeth > lips)
    
    # === 1d Volume Spike (volume > 2.0x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # === 1w Choppiness Index (CHOP > 61.8 = ranging, CHOP < 38.2 = trending) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w).diff()
    tr2 = pd.Series(low_1w).diff().abs()
    tr3 = pd.Series(close_1w).shift(1).diff().abs()
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR(14)
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    sum_atr_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    range_1w = hh_1w - ll_1w
    chop = 100 * np.log10(sum_atr_1w / range_1w) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Regime filters
    ranging_market = chop_aligned > 61.8  # CHOP > 61.8 = ranging (mean revert)
    trending_market = chop_aligned < 38.2  # CHOP < 38.2 = trending (avoid)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(volume_spike[i]) or np.isnan(chop_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_ranging = ranging_market[i]
        is_trending = trending_market[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Alligator lines cross (bullish alignment broken) OR market starts trending
            if not bullish_align[i] or is_trending:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Alligator lines cross (bearish alignment broken) OR market starts trending
            if not bearish_align[i] or is_trending:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bullish Alligator alignment AND volume spike AND ranging market
            if bullish_align[i] and vol_spike and is_ranging:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bearish Alligator alignment AND volume spike AND ranging market
            elif bearish_align[i] and vol_spike and is_ranging:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Williams_Alligator_1dVolumeSpike_1wChop_V1"
timeframe = "12h"
leverage = 1.0