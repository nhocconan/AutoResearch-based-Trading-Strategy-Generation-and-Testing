#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend + volume spike.
# Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3).
# Long when Lips > Teeth > Jaw (bullish alignment) and price > Lips with volume spike in bull trend (close > 1d EMA50).
# Short when Lips < Teeth < Jaw (bearish alignment) and price < Lips with volume spike in bear trend (close < 1d EMA50).
# Alligator identifies trend absence (sleeping) vs presence (awake); EMA50 filters higher-timeframe trend; volume confirms.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
# Works in bull by following Alligator buy signals in uptrend, bear by following sell signals in downtrend.

name = "12h_WilliamsAlligator_1dEMA50_Volume"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe
    def smma(source, period):
        """Smoothed Moving Average (SMMA)"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=np.float64)
        result = np.full_like(source, np.nan, dtype=np.float64)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA(i) = (SMMA(i-1)*(period-1) + source[i]) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Alligator components
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Shift components as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment
        is_bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        is_bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            if is_bullish_alignment and is_bull_trend and close_val > lips_val and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bearish_alignment and is_bear_trend and close_val < lips_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator sleeping (jaw > teeth) or trend reversal
            if jaw_val > teeth_val or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator sleeping (jaw < teeth) or trend reversal
            if jaw_val < teeth_val or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals