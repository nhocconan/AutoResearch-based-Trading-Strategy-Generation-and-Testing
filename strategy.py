#!/usr/bin/env python3
"""
1d_Williams_Alligator_ElderRay_Trend
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) for trend direction + Elder Ray (Bear/Bull Power) for momentum confirmation.
- Long when: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND price > Jaw
- Short when: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND price < Jaw
- Exit when: Alligator alignment breaks (Lips crosses Teeth) OR price crosses Jaw in opposite direction
Uses 1d timeframe with 1h Alligator/Elder Ray for timely signals while maintaining low trade frequency.
Targets 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
Works in both bull (trend following) and bear (counter-trend reversals at extremes) markets.
"""

name = "1d_Williams_Alligator_ElderRay_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1h data for Alligator and Elder Ray calculations
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 13:
        return np.zeros(n)
    
    # 1d OHLCV
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    
    # --- Williams Alligator: SMoothed Medians ---
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(values, period):
        """Smoothed Moving Average"""
        sma = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return sma
        # First value is simple average
        sma[period-1] = np.mean(values[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(values)):
            sma[i] = (sma[i-1] * (period-1) + values[i]) / period
        return sma
    
    # Calculate SMMA for each component
    jaw_raw = smma(median_1h := (df_1h['high'].values + df_1h['low'].values) / 2, 13)
    teeth_raw = smma(median_1h, 8)
    lips_raw = smma(median_1h, 5)
    
    # Apply shifts (Jaw +8, Teeth +5, Lips +3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Align to 1d timeframe
    jaw_1d = align_htf_to_ltf(prices, df_1h, jaw)
    teeth_1d = align_htf_to_ltf(prices, df_1h, teeth)
    lips_1d = align_htf_to_ltf(prices, df_1h, lips)
    
    # --- Elder Ray: Bull Power and Bear Power ---
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13_1h = pd.Series(df_1h['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1h_aligned = align_htf_to_ltf(prices, df_1h, ema13_1h)
    
    bull_power = high_1d - ema13_1h_aligned
    bear_power = low_1d - ema13_1h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20  # for Alligator shifts and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw_1d[i]) or np.isnan(teeth_1d[i]) or np.isnan(lips_1d[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Alligator alignment
        bullish_alignment = lips_1d[i] > teeth_1d[i] > jaw_1d[i]
        bearish_alignment = lips_1d[i] < teeth_1d[i] < jaw_1d[i]
        
        # Elder Ray conditions
        bull_power_pos = bull_power[i] > 0
        bear_power_neg = bear_power[i] < 0
        
        # Price relative to Jaw
        price_above_jaw = close_1d[i] > jaw_1d[i]
        price_below_jaw = close_1d[i] < jaw_1d[i]
        
        if position == 0:
            # Look for entries
            if bullish_alignment and bull_power_pos and price_above_jaw:
                # Long: bullish alignment + positive bull power + price above jaw
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and bear_power_neg and price_below_jaw:
                # Short: bearish alignment + negative bear power + price below jaw
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: alignment breaks OR price crosses below jaw
                if not bullish_alignment or price_below_jaw:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: alignment breaks OR price crosses above jaw
                if not bearish_alignment or price_above_jaw:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals