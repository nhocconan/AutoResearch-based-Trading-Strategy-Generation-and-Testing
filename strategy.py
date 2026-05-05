#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Chop Regime Filter
# Long when: Jaw < Teeth < Lips (bullish alignment) AND 12h volume > 1.5 * 20-period avg AND Chop(14) > 61.8 (ranging market)
# Short when: Jaw > Teeth > Lips (bearish alignment) AND 12h volume > 1.5 * 20-period avg AND Chop(14) > 61.8 (ranging market)
# Exit when Alligator alignment breaks or Chop < 38.2 (trending market)
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-30 trades/year per symbol.
# Williams Alligator identifies trend absence (ideal for ranging markets), volume spike confirms participation,
# Chop filter ensures we only trade in ranging regimes where mean reversion works best.
# Works in bull markets via longs during bull ranges and bear markets via shorts during bear ranges.

name = "12h_WilliamsAlligator_VolumeSpike_CHOP_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Alligator and volume calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA = smoothed moving average (similar to EMA but with different alpha)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_12h = df_12h['close'].values
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Shift Jaw forward 8, Teeth forward 5, Lips forward 3
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Invalidate shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to prices timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 12h volume spike filter
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20 = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Calculate Chopiness Index (CHOP) on 12h
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period:
            return np.full_like(high_arr, np.nan, dtype=float)
        
        # True Range
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TR over period
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        # Highest high and lowest low over period
        hh = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        
        # Chop = LOG10(sum(tr)/(hh-ll)) / LOG10(period) * 100
        # Avoid division by zero
        hh_ll = hh - ll
        chop = np.full_like(high_arr, np.nan, dtype=float)
        mask = (hh_ll > 0) & ~np.isnan(tr_sum)
        chop[mask] = (np.log10(tr_sum[mask] / hh_ll[mask]) / np.log10(period)) * 100
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_align = chop  # Already on prices timeframe
    
    # Regime filters: Chop > 61.8 = ranging (good for mean reversion), Chop < 38.2 = trending
    chop_ranging = chop_align > 61.8
    chop_trending = chop_align < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(chop_ranging[i]) or np.isnan(chop_trending[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: Jaw < Teeth < Lips
            bullish = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
            # Bearish Alligator alignment: Jaw > Teeth > Lips
            bearish = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
            
            # Long conditions: bullish alignment AND volume spike AND ranging market
            if bullish and volume_spike[i] and chop_ranging[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment AND volume spike AND ranging market
            elif bearish and volume_spike[i] and chop_ranging[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR market starts trending
            bullish = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
            if (not bullish) or chop_trending[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR market starts trending
            bearish = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
            if (not bearish) or chop_trending[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals