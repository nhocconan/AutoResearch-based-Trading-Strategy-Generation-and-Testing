#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
- Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND 1w EMA50 uptrend AND volume > 1.5x 20-period average
- Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND 1w EMA50 downtrend AND volume > 1.5x 20-period average
- Exit when Alligator alignment breaks (jaws-teeth-lips not in proper order) OR price crosses 8-period EMA
- Uses 1w EMA50 for strong HTF trend alignment to avoid counter-trend entries
- Williams Alligator catches trends early with smoothing to reduce whipsaws
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Alligator: three smoothed moving averages
    # Jaw: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    # SMMA = smoothed moving average (similar to EMA but different smoothing)
    
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # First 8 values of jaw, 5 of teeth, 3 of lips are invalid due to shift
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 8-period EMA for exit signal
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 20)  # Need 50 for 1w EMA50, 13 for jaw, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(ema8[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i]
        bearish_alignment = jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]
        
        # Trend filter (using 1w EMA50)
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Price relative to lips
        price_above_lips = close[i] > lips_shifted[i]
        price_below_lips = close[i] < lips_shifted[i]
        
        if position == 0:
            # Long: Bullish alignment + price above lips + uptrend + volume confirmation
            if bullish_alignment and price_above_lips and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price below lips + downtrend + volume confirmation
            elif bearish_alignment and price_below_lips and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Exit 1: Alligator alignment breaks
            if position == 1:
                # Exit long if bullish alignment breaks
                if not (jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short if bearish alignment breaks
                if not (jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]):
                    exit_signal = True
            
            # Exit 2: Price crosses 8-period EMA (mean reversion)
            if not exit_signal:
                if position == 1 and close[i] < ema8[i]:
                    exit_signal = True
                elif position == -1 and close[i] > ema8[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0