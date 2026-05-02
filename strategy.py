#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Uses 12h timeframe for signal generation (Williams Alligator jaw/teeth/lips crossover)
# Uses 1d EMA34 for trend filter (price > EMA34 = bullish bias, price < EMA34 = bearish bias)
# Volume confirmation (2.0x 24-period average on 12h) ensures institutional participation
# Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) smoothed with SMMA
# Long: Lips > Teeth > Jaw + price > 1d EMA34 + volume confirm
# Short: Lips < Teeth < Jaw + price < 1d EMA34 + volume confirm
# Exit: Alligator lines re-cross (Lips-Teeth or Teeth-Jaw crossover)
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag
# Works in bull markets via trend-following crossovers, in bear via filtered signals avoiding whipsaws

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
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
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current value) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set shifted values to NaN where shift creates invalid data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation (2.0x 24-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator: Lips > Teeth > Jaw
            bullish_alligator = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
            # Bearish Alligator: Lips < Teeth < Jaw
            bearish_alligator = lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i]
            
            # Long: Bullish Alligator + price > 1d EMA34 + volume confirm
            if bullish_alligator and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + price < 1d EMA34 + volume confirm
            elif bearish_alligator and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator lines re-cross (Lips-Teeth crossover downward)
            if lips_shifted[i] < teeth_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator lines re-cross (Lips-Teeth crossover upward)
            if lips_shifted[i] > teeth_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals