#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator + 1d EMA trend filter + volume confirmation
    # Long: Jaw < Teeth < Lips (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x 20-period average
    # Short: Jaw > Teeth > Lips (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x 20-period average
    # Exit: Alligator lines cross (Jaw-Teeth or Teeth-Lips) indicating trend weakness
    # Using 1d for EMA50 trend filter, 12h for Alligator and volume
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 12-37 trades/year (~50-150 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams Alligator on 12h (Smoothed Medians with specific periods)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(values, period):
        """Smoothed Moving Average"""
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply shifts (Alligator specific)
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 1d EMA50, only short if price < 1d EMA50
        long_trend_ok = close[i] > ema_1d_aligned[i]
        short_trend_ok = close[i] < ema_1d_aligned[i]
        
        # Alligator alignment: Jaw < Teeth < Lips = bullish, Jaw > Teeth > Lips = bearish
        bullish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Entry logic: Alligator alignment + volume + trend
        long_entry = bullish_alignment and vol_confirm and long_trend_ok
        short_entry = bearish_alignment and vol_confirm and short_trend_ok
        
        # Exit logic: Alligator lines cross (trend weakening)
        jaw_teeth_cross = (jaw[i] > teeth[i]) and (jaw[i-1] <= teeth[i-1])  # Bullish to bearish
        teeth_lips_cross = (teeth[i] > lips[i]) and (teeth[i-1] <= lips[i-1])  # Bullish to bearish
        long_exit = jaw_teeth_cross or teeth_lips_cross
        
        jaw_teeth_cross_bear = (jaw[i] < teeth[i]) and (jaw[i-1] >= teeth[i-1])  # Bearish to bullish
        teeth_lips_cross_bear = (teeth[i] < lips[i]) and (teeth[i-1] >= lips[i-1])  # Bearish to bullish
        short_exit = jaw_teeth_cross_bear or teeth_lips_cross_bear
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_williams_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0