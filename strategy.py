#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Long when Alligator Jaw < Teeth < Lips (bullish alignment) AND price > Lips AND close > 1d EMA50 AND volume > 1.8x 20-period average.
# Short when Alligator Jaw > Teeth > Lips (bearish alignment) AND price < Lips AND close < 1d EMA50 AND volume > 1.8x 20-period average.
# Uses ATR(14) trailing stop (2.0x) for risk control.
# Williams Alligator identifies trend initiation/continuation with smoothing to reduce whipsaw.
# Volume confirmation filters breakouts; EMA50 on 1d avoids counter-trend trades.
# Discrete position sizing (0.25) minimizes fee churn. Target: 80-160 total trades over 4 years (20-40/year) on 4h.

name = "4h_WilliamsAlligator_1dEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator (13,8,5) with SMMA (Smoothed Moving Average)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate SMMA: smma[i] = (smma[i-1] * (period-1) + close[i]) / period
    def smma(source, period):
        result = np.full_like(source, np.nan)
        if len(source) >= period:
            result[period-1] = np.mean(source[:period])
            for i in range(period, len(source)):
                result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    
    # Shift jaw and teeth forward to align with predictive nature
    jaw_shifted = np.roll(jaw, -jaw_shift)
    teeth_shifted = np.roll(teeth, -teeth_shift)
    # lips typically not shifted or shifted less; using unshifted lips
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to LTF (4h)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish alignment (jaw < teeth < lips) AND price > lips AND close > EMA50 AND volume confirmation
            if (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and 
                close[i] > lips_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Bearish alignment (jaw > teeth > lips) AND price < lips AND close < EMA50 AND volume confirmation
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                  close[i] < lips_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals