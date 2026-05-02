#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + 1d EMA50 trend filter with volume confirmation
# Williams Alligator identifies trend via jaw/teeth/lips alignment. Elder Ray measures bull/bear power.
# Combined with 1d EMA50 trend and volume spike, this captures strong trends while filtering chop.
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years).
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
# Works in bull markets (bullish alignment + positive Elder Ray) and bear markets (bearish alignment + negative Elder Ray).

name = "12h_WilliamsAlligator_ElderRay_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 1d EMA50 calculation
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (smoothed medians)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Invalidate the shifted values that don't have enough history
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Elder Ray Index (requires 13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.8 * vol_ema_20)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 80
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator alignment
        # Bullish: Lips > Teeth > Jaw (all ascending)
        # Bearish: Jaw > Teeth > Lips (all descending)
        bullish_align = (lips_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > jaw_shifted[i])
        bearish_align = (jaw_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > lips_shifted[i])
        
        # Determine trend bias from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment + positive Elder Ray + volume confirmation + uptrend
            if bullish_align and (bull_power[i] > 0) and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + negative Elder Ray + volume confirmation + downtrend
            elif bearish_align and (bear_power[i] < 0) and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish Alligator alignment OR Elder Ray turns negative OR trend changes
            if bearish_align or (bull_power[i] < 0) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish Alligator alignment OR Elder Ray turns positive OR trend changes
            if bullish_align or (bear_power[i] > 0) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals