#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator (SMMA13,8,5) with 1-day EMA(50) trend filter and volume confirmation
# Williams Alligator provides trend direction via jaw-teeth-lips alignment
# EMA(50) on daily timeframe filters for higher timeframe trend alignment
# Volume confirmation ensures breakouts have institutional participation
# Designed to work in both bull and bear markets by only trading in direction of higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_williams_alligator_1d_ema50_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_daily = df_daily['close'].values
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Williams Alligator components (SMMA: Smoothed Moving Average)
    # Jaw: SMMA(13, 8) - 13-period smoothed, shifted 8 bars forward
    # Teeth: SMMA(8, 5) - 8-period smoothed, shifted 5 bars forward
    # Lips: SMMA(5, 3) - 5-period smoothed, shifted 3 bars forward
    
    def smma(data, period, shift):
        """Smoothed Moving Average with shift"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        # Calculate SMA for first value
        sma = np.full_like(data, np.nan)
        sma[period-1] = np.mean(data[:period])
        # Calculate SMMA for subsequent values
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        # Apply shift (forward shift means we use future values, so we need to shift back)
        # For Alligator, we shift the smoothed averages forward, so we lag them in our calculation
        shifted = np.full_like(sma, np.nan)
        if shift < len(sma):
            shifted[shift:] = sma[:-shift]
        return shifted
    
    jaw = smma(close, 13, 8)
    teeth = smma(close, 8, 5)
    lips = smma(close, 5, 3)
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_daily_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Alligator lines reverse (jaws below teeth) or trend changes
            if jaw[i] < teeth[i] or close[i] < ema_50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Alligator lines reverse (jaws above teeth) or trend changes
            if jaw[i] > teeth[i] or close[i] > ema_50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Alligator alignment: jaws > teeth > lips = uptrend, jaws < teeth < lips = downtrend
            # Add small epsilon to avoid floating point equality issues
            epsilon = 1e-10
            jaw_above_teeth = jaw[i] > teeth[i] + epsilon
            teeth_above_lips = teeth[i] > lips[i] + epsilon
            jaw_below_teeth = jaw[i] < teeth[i] - epsilon
            teeth_below_lips = teeth[i] < lips[i] - epsilon
            
            uptrend_aligned = jaw_above_teeth and teeth_above_lips
            downtrend_aligned = jaw_below_teeth and teeth_below_lips
            
            # Higher timeframe trend filter: price above/below daily EMA(50)
            htf_uptrend = close[i] > ema_50_daily_aligned[i]
            htf_downtrend = close[i] < ema_50_daily_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * average volume
            volume_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long: Alligator aligned up + HTF uptrend + volume
            if uptrend_aligned and htf_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Alligator aligned down + HTF downtrend + volume
            elif downtrend_aligned and htf_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals