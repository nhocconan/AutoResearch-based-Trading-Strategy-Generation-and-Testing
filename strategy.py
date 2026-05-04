#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Choppiness Regime Filter
# Long when: Alligator bullish alignment (jaw < teeth < lips) AND 1d volume > 2.0x 20-period EMA AND Choppiness Index > 61.8 (range regime)
# Short when: Alligator bearish alignment (jaw > teeth > lips) AND 1d volume > 2.0x 20-period EMA AND Choppiness Index > 61.8 (range regime)
# Uses Williams Alligator (13,8,5 SMAs with future shift) for trend detection, volume spike for momentum confirmation, and Chop filter to avoid false signals in strong trends.
# Designed for 12h timeframe: targets 12-37 trades/year (50-150 total over 4 years) with discrete position sizing (0.25) to minimize fee drag.
# Works in ranging markets via mean-reversion at Alligator extremes and avoids trending markets where Alligator whipsaws.

name = "12h_WilliamsAlligator_1dVolSpike_ChopFilter"
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
    
    # Get 1d data for HTF filters - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume EMA20 for spike filter
    vol_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ema_20_1d * 2.0)  # Volume at least 2.0x average for spike
    
    # Align 1d volume spike to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate 1d Choppiness Index (14-period)
    # Chop = 100 * log10(sum(ATR(14)) / (log10(HHV(14) - LLV(14)) / log10(14)))
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).sum().values
    hhvl_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values - \
              pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_14 / (hhvl_14 / np.log10(14)))
    chop_1d = np.where(hhvl_14 > 0, chop_1d, 50.0)  # Avoid division by zero
    
    # Choppiness regime: > 61.8 = range (favorable for mean reversion)
    chop_regime_1d = chop_1d > 61.8
    
    # Align 1d filters to 12h timeframe
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d.astype(float))
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average (similar to RMA/Wilder's)"""
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is SMA
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply Alligator shifts (jaw shifted 8, teeth shifted 5, lips shifted 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that went out of bounds
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Alligator alignments
    alligator_bullish = (jaw_shifted < teeth_shifted) & (teeth_shifted < lips_shifted)
    alligator_bearish = (jaw_shifted > teeth_shifted) & (teeth_shifted > lips_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish AND volume spike AND chop regime (range)
            if (alligator_bullish[i] and 
                volume_spike_aligned[i] > 0.5 and 
                chop_regime_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish AND volume spike AND chop regime (range)
            elif (alligator_bearish[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  chop_regime_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR chop regime ends (trend begins)
            if (alligator_bearish[i] > 0.5 or 
                chop_regime_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR chop regime ends (trend begins)
            if (alligator_bullish[i] > 0.5 or 
                chop_regime_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals