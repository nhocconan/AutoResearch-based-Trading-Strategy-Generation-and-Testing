#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d volume confirmation and 1w trend filter
# - Williams Alligator (5,3,0) / (8,5,0) / (13,8,0) SMAs on 4h for trend direction
# - Long when jaw < teeth < lips (bullish alignment) + price > lips
# - Short when jaw > teeth > lips (bearish alignment) + price < lips
# - 1d volume > 1.8x 20-period average for conviction (avoid low-conviction moves)
# - 1w EMA(50) filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - Entry only during 08:00-20:00 UTC session to avoid low-volume periods
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to catch trends in both bull and bear markets via multi-timeframe alignment
# - Target: 20-40 trades/year to minimize fee drag while capturing strong moves

name = "4h_WilliamsAlligator_1dVolume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Alligator
    df_4h = get_htf_data(prices, '4h')
    
    # Williams Alligator on 4h
    # Jaw: 13-period SMMA, 8-period shift
    # Teeth: 8-period SMMA, 5-period shift  
    # Lips: 5-period SMMA, 3-period shift
    close_4h = df_4h['close'].values
    
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean()
        # Initialize first value as SMA
        result = np.full_like(series, np.nan, dtype=float)
        if len(sma) >= period:
            result[period-1] = sma.iloc[period-1]
            for i in range(period, len(series)):
                result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw_raw = smma(close_4h, 13)
    teeth_raw = smma(close_4h, 8)
    lips_raw = smma(close_4h, 5)
    
    # Apply shifts: jaw 8, teeth 5, lips 3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set invalid values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute session filter (08:00-20:00 UTC)
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
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.8x average (need actual 1d volume)
        # Get the actual 1d volume aligned to current 4h bar
        idx_1d = i // (24*60//4)  # 4h to 1d: 6 bars per day
        if idx_1d < len(vol_1d):
            vol_current = vol_1d[idx_1d]
            vol_average = vol_ma_1d[idx_1d] if not np.isnan(vol_ma_1d[idx_1d]) else 0
            volume_filter = vol_average > 0 and vol_current > 1.8 * vol_average
        else:
            volume_filter = False
        
        if position == 0:
            # Bullish Alligator alignment: jaw < teeth < lips
            bullish = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
            # Bearish Alligator alignment: jaw > teeth > lips
            bearish = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
            
            # Look for long entry: bullish alignment + price > lips + above weekly EMA + volume
            if (bullish and close[i] > lips_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: bearish alignment + price < lips + below weekly EMA + volume
            elif (bearish and close[i] < lips_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on bearish alignment or price below lips or below weekly EMA
            bearish = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
            if bearish or close[i] < lips_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on bullish alignment or price above lips or above weekly EMA
            bullish = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
            if bullish or close[i] > lips_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals