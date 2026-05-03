#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume spike confirmation.
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# In trending markets: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish).
# In ranging markets: lines intertwine. We enter only when Alligator is "awake" (trending)
# and aligned with 1d EMA34, confirmed by volume spike.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
# Works in both bull and bear: Alligator identifies trend direction, volume confirms strength,
# 1d EMA34 filter avoids counter-trend trades during major reversals.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 6h data (primary timeframe)
    # Jaw: Blue line - 13-period SMMA shifted 8 bars
    # Teeth: Red line - 8-period SMMA shifted 5 bars  
    # Lips: Green line - 5-period SMMA shifted 3 bars
    def smma(values, period):
        """Smoothed Moving Average"""
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with NaN
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 1d EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(30) for stoploss (using 6h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=30, min_periods=30, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 30-bar average (on 6h data)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Alligator conditions
        # Bullish: Lips > Teeth > Jaw (alligator awake, eating up)
        bullish_alligator = (lips_val > teeth_val) and (teeth_val > jaw_val)
        # Bearish: Lips < Teeth < Jaw (alligator awake, eating down)
        bearish_alligator = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Entry conditions
        # Long: bullish alligator + price above 1d EMA34 + volume spike
        long_entry = bullish_alligator and (close[i] > ema_trend) and vol_spike
        # Short: bearish alligator + price below 1d EMA34 + volume spike
        short_entry = bearish_alligator and (close[i] < ema_trend) and vol_spike
        
        # Exit conditions (trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals