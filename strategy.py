#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike with 1d EMA34 trend filter.
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND Elder Bull Power > 0 AND 4h volume > 2.0x 20-period volume MA AND 1d close > 1d EMA34.
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Elder Bear Power < 0 AND 4h volume > 2.0x 20-period volume MA AND 1d close < 1d EMA34.
# Exit on Alligator crossover (jaws crosses teeth) or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Williams Alligator identifies trend via smoothed medians, Elder Ray measures bull/bear power behind move, volume confirms participation.
# Works in both bull and bear markets by trading in direction of 1d EMA34 trend when Alligator aligns and volume spikes.
# Target: 75-200 total trades over 4 years (19-50/year) via tight entry conditions.

name = "4h_WilliamsAlligator_ElderRay_VolumeSpike_1dEMA34_Session"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator (SMMA = Smoothed Moving Average)
    # Jaw: SMMA(median, 13, 8) - 8 period forward shift
    # Teeth: SMMA(median, 8, 5) - 5 period forward shift  
    # Lips: SMMA(median, 5, 3) - 3 period forward shift
    median = (high + low) / 2.0
    
    def smma(data, period, shift):
        """Smoothed Moving Average with forward shift"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        # Calculate initial SMA
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        # Smooth: SMMA(t) = (SMMA(t-1)*(period-1) + data(t)) / period
        smma_vals = np.full_like(data, np.nan)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(data)):
            if not np.isnan(smma_vals[i-1]):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + data[i]) / period
            else:
                smma_vals[i] = np.nan
        # Apply forward shift
        shifted = np.full_like(data, np.nan)
        if shift < len(data):
            shifted[shift:] = smma_vals[:-shift] if shift > 0 else smma_vals
        return shifted
    
    jaw = smma(median, 13, 8)
    teeth = smma(median, 8, 5)
    lips = smma(median, 5, 3)
    
    # Align Alligator components to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_ma_4h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Alligator alignment conditions
        bullish_alignment = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # 1d trend conditions
        trend_up = close[i] > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close[i] < ema_34_1d_aligned[i]  # 1d downtrend
        
        # Volume spike condition: current 4h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_4h[i] * 2.0)
        
        # Elder Ray conditions
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        if position == 0:
            # Long: Bullish Alligator AND Elder Bull Power > 0 AND volume spike AND 1d uptrend AND session
            if bullish_alignment and bull_power_positive and volume_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND Elder Bear Power < 0 AND volume spike AND 1d downtrend AND session
            elif bearish_alignment and bear_power_negative and volume_spike and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator crossover (jaw crosses teeth) OR trend changes
            jaw_cross_teeth = jaw_aligned[i] > teeth_aligned[i]
            if jaw_cross_teeth or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator crossover (jaw crosses teeth) OR trend changes
            jaw_cross_teeth = jaw_aligned[i] < teeth_aligned[i]
            if jaw_cross_teeth or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals