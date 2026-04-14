#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Long when Alligator jaws (blue line) cross above teeth (red line) with price above 1d EMA50 and volume > 1.5x 20-period average
# Short when jaws cross below teeth with price below 1d EMA50 and volume > 1.5x 20-period average
# Exit when jaws cross back in opposite direction or price crosses 1d EMA50
# Williams Alligator uses SMAs of 13, 8, 5 periods with future shifts (8, 5, 3) to avoid look-ahead
# Designed to catch trends in both bull and bear markets by aligning with higher timeframe trend
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components on 4h data
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    # SMMA = Smoothed Moving Average (similar to EMA but with different smoothing)
    
    # Calculate SMMA using EMA as approximation (close enough for trading purposes)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Typical price for Alligator calculation
    typical_price_4h = (high_4h + low_4h + close_4h) / 3
    
    # Jaw (13-period EMA, shifted 8)
    jaw = pd.Series(typical_price_4h).ewm(span=13, adjust=False).mean().values
    # Teeth (8-period EMA, shifted 5)
    teeth = pd.Series(typical_price_4h).ewm(span=8, adjust=False).mean().values
    # Lips (5-period EMA, shifted 3)
    lips = pd.Series(typical_price_4h).ewm(span=5, adjust=False).mean().values
    
    # Apply shifts (to avoid look-ahead, we use already-shifted values)
    # Since we're calculating on historical data, we shift the values forward in time
    # Jaw shifted 8 bars: current jaw value actually occurred 8 bars ago
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan  # First 8 values are invalid
    
    # Teeth shifted 5 bars
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan
    
    # Lips shifted 3 bars
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips_shifted)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need at least 50 for EMA50)
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]
        
        # Williams Alligator signals:
        # Jaw above Teeth = bullish (green/red/blue alignment from bottom to top)
        # Jaw below Teeth = bearish
        
        if position == 0:
            # Long setup: Jaw crosses above Teeth with price above 1d EMA50 and volume confirmation
            jaw_above_teeth = jaw_aligned[i] > teeth_aligned[i]
            jaw_was_below_teeth = jaw_aligned[i-1] <= teeth_aligned[i-1] if i > 0 else False
            bullish_cross = jaw_above_teeth and jaw_was_below_teeth
            
            if (bullish_cross and 
                price > ema_50_1d_aligned[i] and 
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: Jaw crosses below Teeth with price below 1d EMA50 and volume confirmation
            elif (jaw_aligned[i] < teeth_aligned[i] and 
                  jaw_aligned[i-1] >= teeth_aligned[i-1] if i > 0 else False and  # bearish cross
                  price < ema_50_1d_aligned[i] and 
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Jaw crosses back below Teeth OR price crosses below 1d EMA50
            jaw_below_teeth = jaw_aligned[i] < teeth_aligned[i]
            jaw_was_above_teeth = jaw_aligned[i-1] >= teeth_aligned[i-1] if i > 0 else False
            bearish_cross = jaw_below_teeth and jaw_was_above_teeth
            
            if bearish_cross or price < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Jaw crosses back above Teeth OR price crosses above 1d EMA50
            jaw_above_teeth = jaw_aligned[i] > teeth_aligned[i]
            jaw_was_below_teeth = jaw_aligned[i-1] <= teeth_aligned[i-1] if i > 0 else False
            bullish_cross = jaw_above_teeth and jaw_was_below_teeth
            
            if bullish_cross or price > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0