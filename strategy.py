#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour strategy using 1-day Williams Alligator and Elder Ray Index with 1-week ADX filter.
# Williams Alligator (13,8,5 SMAs) identifies trend direction and absence (all lines intertwined = no trend).
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
# ADX(10) from 1-week filters for strong trends (ADX > 25) to avoid whipsaws in ranging markets.
# Entry: Go long when Bull Power > 0 and Alligator jaws < teeth < lips (bullish alignment).
# Entry: Go short when Bear Power > 0 and Alligator jaws > teeth > lips (bearish alignment).
# Exit: When opposing power becomes positive or Alligator re-aligns against position.
# Volume confirmation: Current volume > 1.5x 20-period average to confirm participation.
# Position size: 0.25 (25%) to balance return and drawdown.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    
    # Load 1w data ONCE for ADX filter
    df_1w = get_htf_data(prices, '1w')
    
    # Williams Alligator on 1d: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    if len(df_1d) < 13:
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    if len(df_1d) < 13:
        return np.zeros(n)
    
    ema13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13
    bear_power = ema13 - df_1d['low'].values
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # ADX(10) on 1w for trend strength filter
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=10, adjust=False, min_periods=10).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1w = pd.Series(dx).ewm(span=10, adjust=False, min_periods=10).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(13, 10, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 = strong trend
        strong_trend = adx_1w_aligned[i] > 25
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Alligator alignment checks
        bullish_alignment = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        if position == 0:
            if strong_trend and volume_confirmed:
                # Long: Bull Power > 0 and bullish Alligator alignment
                if (bull_power_aligned[i] > 0 and 
                    bullish_alignment):
                    position = 1
                    signals[i] = position_size
                # Short: Bear Power > 0 and bearish Alligator alignment
                elif (bear_power_aligned[i] > 0 and 
                      bearish_alignment):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                # In weak trend, no new entries
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bear Power becomes positive or Alligator loses bullish alignment
            if (bear_power_aligned[i] > 0 or 
                not bullish_alignment):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bull Power becomes positive or Alligator loses bearish alignment
            if (bull_power_aligned[i] > 0 or 
                not bearish_alignment):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Alligator_ElderRay_1wADX_v1"
timeframe = "12h"
leverage = 1.0