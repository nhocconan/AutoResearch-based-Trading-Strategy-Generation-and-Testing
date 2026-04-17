#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Alligator Jaw AND volume > 1.5x 20-period average AND price > 1w EMA50.
Short when price breaks below Alligator Jaw AND volume > 1.5x 20-period average AND price < 1w EMA50.
Exit when price crosses the Alligator Jaw in opposite direction.
Williams Alligator (Jaw=TEETH=LIPS SMMA) provides dynamic support/resistance, 1w EMA50 filters for
long-term trend, volume confirmation reduces false breakouts. Designed for low trade frequency
(12-37/year) to minimize fee drag while capturing strong breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA) aka Wilder's MA"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    sma = np.mean(series[:period])
    result = np.full_like(series, np.nan, dtype=float)
    result[period-1] = sma
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator on 12h timeframe
    # Typical Price = (high + low + close) / 3
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    
    # Jaw: SMMA(TP, 13, 8) - 8 bars ahead
    jaw_raw = smma(typical_price_12h, 13)
    jaw_12h = np.roll(jaw_raw, 8)  # shift 8 bars ahead
    
    # Teeth: SMMA(TP, 8, 5) - 5 bars ahead
    teeth_raw = smma(typical_price_12h, 8)
    teeth_12h = np.roll(teeth_raw, 5)  # shift 5 bars ahead
    
    # Lips: SMMA(TP, 5, 3) - 3 bars ahead
    lips_raw = smma(typical_price_12h, 5)
    lips_12h = np.roll(lips_raw, 3)  # shift 3 bars ahead
    
    # Calculate EMA50 on 1w timeframe
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe (which is our primary timeframe)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # For Alligator, we use Jaw as the main support/resistance line
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw = jaw_aligned[i]
        ema_50 = ema_50_1w_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above Jaw AND volume > 1.5x avg AND price > 1w EMA50 (bullish trend)
            if high_price > jaw and vol > 1.5 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Jaw AND volume > 1.5x avg AND price < 1w EMA50 (bearish trend)
            elif low_price < jaw and vol > 1.5 * vol_ma and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Jaw
            if price < jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Jaw
            if price > jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_JawBreakout_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0