#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Uses 12h timeframe for signal generation (Williams Alligator crossover)
# 1d EMA34 for trend filter (only trade in direction of daily trend)
# Volume confirmation (2.0x 24-period average) ensures institutional participation
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Williams Alligator catches trends early with smoothed median price (Jaw/Teeth/Lips)
# Works in bull markets via trend-following crossovers, in bear via daily trend filter avoiding counter-trend trades
# Designed for low trade frequency to minimize fee drag (critical for 12h timeframe)

name = "12h_WilliamsAlligator_1dEMA34_Trend_Volume_v1"
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
    
    # Load 12h data ONCE before loop for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    # Typical Price = (H + L + C)/3
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    
    # Jaw (Blue Line): 13-period SMMA, shifted 8 bars forward
    jaw_12h = pd.Series(typical_price_12h).rolling(window=13, min_periods=13).mean().values
    jaw_12h = np.roll(jaw_12h, 8)
    jaw_12h[:8] = np.nan
    
    # Teeth (Red Line): 8-period SMMA, shifted 5 bars forward
    teeth_12h = pd.Series(typical_price_12h).rolling(window=8, min_periods=8).mean().values
    teeth_12h = np.roll(teeth_12h, 5)
    teeth_12h[:5] = np.nan
    
    # Lips (Green Line): 5-period SMMA, shifted 3 bars forward
    lips_12h = pd.Series(typical_price_12h).rolling(window=5, min_periods=5).mean().values
    lips_12h = np.roll(lips_12h, 3)
    lips_12h[:3] = np.nan
    
    # Align Alligator lines to lower timeframe (1h)
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (2.0x 24-period average on 1h)
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
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Lips > Teeth > Jaw (Alligator bullish alignment) + price > 1d EMA34 + volume confirm
            if (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (Alligator bearish alignment) + price < 1d EMA34 + volume confirm
            elif (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator lines cross back (Lips < Teeth) or strong reversal
            if lips_12h_aligned[i] < teeth_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator lines cross back (Lips > Teeth) or strong reversal
            if lips_12h_aligned[i] > teeth_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals