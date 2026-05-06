#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Long when Alligator jaws (13) < teeth (8) < lips (5) AND 1d close > 1d EMA50 AND volume > 2.0 * 20-bar avg volume
# Short when Alligator jaws (13) > teeth (8) > lips (5) AND 1d close < 1d EMA50 AND volume > 2.0 * 20-bar avg volume
# Exit when Alligator lines cross (jaws == teeth) or opposite signal appears
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator identifies trend alignment, EMA50 provides multi-timeframe trend filter
# High volume threshold (2.0x) reduces false signals and controls trade frequency
# Strategy works in both bull (trend following) and bear (counter-trend via Alligator alignment) markets

name = "12h_WilliamsAlligator_1dEMA50_Volume_v1"
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
    
    # Calculate Williams Alligator (SMMA = Smoothed Moving Average)
    # Jaws: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Value) / Period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaws = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply shifts (Alligator specific)
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaws_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator signals with trend and volume filters
            # Long: Jaws < Teeth < Lips (bullish alignment) AND uptrend AND volume spike
            if jaws_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < lips_shifted[i] and \
               close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) AND downtrend AND volume spike
            elif jaws_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > lips_shifted[i] and \
                 close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines cross (jaws >= teeth) or opposite signal
            if jaws_shifted[i] >= teeth_shifted[i] or \
               (jaws_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > lips_shifted[i] and \
                close[i] < ema50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines cross (jaws <= teeth) or opposite signal
            if jaws_shifted[i] <= teeth_shifted[i] or \
               (jaws_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < lips_shifted[i] and \
                close[i] > ema50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals