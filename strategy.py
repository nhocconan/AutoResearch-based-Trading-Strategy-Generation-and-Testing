#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and 1w trend filter
# - Williams Alligator (Jaw=13, Teeth=8, Lips=5) defines trend: 
#   * Bullish: Lips > Teeth > Jaw (green alignment)
#   * Bearish: Jaw > Teeth > Lips (red alignment)
# - 1d volume > 1.3x 20-period average for confirmation
# - 1w close > 1w EMA(50) for bull regime, < for bear regime
# - Entry on Alligator alignment in direction of weekly trend
# - Exit on opposing Alligator alignment or weekly trend reversal
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to capture trends while avoiding whipsaws in ranging markets
# - Target: 15-35 trades/year to stay within fee limits

name = "12h_WilliamsAlligator_1dVolume_1wTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend regime
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 12h data (Smoothed Medians)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smoothed_ma(data, period):
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean()
        # SMMA: smoothed moving average ( Wilder's smoothing)
        smma = np.full_like(sma, np.nan, dtype=float)
        if len(sma) >= period:
            smma[period-1] = sma[period-1]
            for i in range(period, len(sma)):
                smma[i] = (smma[i-1] * (period-1) + sma[i]) / period
        return smma
    
    jaw = smoothed_ma(close, 13)
    teeth = smoothed_ma(close, 8)
    lips = smoothed_ma(close, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set shifted values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Pre-compute session filter (08:00-20:00 UTC) - less critical for 12h but keeps consistency
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.3x 1d average (approximation)
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.3 * vol_ma_1d_aligned[i]
        
        # Alligator alignments
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Weekly trend filter
        weekly_bull = close[i] > ema_50_1w_aligned[i]
        weekly_bear = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for long entry: bullish Alligator alignment + weekly uptrend + volume
            if bullish_alignment and weekly_bull and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: bearish Alligator alignment + weekly downtrend + volume
            elif bearish_alignment and weekly_bear and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on bearish Alligator alignment or weekly trend reversal
            if bearish_alignment or not weekly_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on bullish Alligator alignment or weekly trend reversal
            if bullish_alignment or not weekly_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals