#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend filter + volume spike confirmation
# Uses 12h timeframe for signal generation with Williams Alligator (JAWS/TEETH/LIPS)
# 1d EMA34 for trend filter (price > EMA34 = bullish bias, price < EMA34 = bearish bias)
# Volume confirmation (2.0x 24-period average on 12h) ensures institutional participation
# Williams Alligator provides clear trend signals: LIPS crosses above TEETH = long signal
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag
# Works in bull markets via trend-following Alligator signals, in bear via EMA34 filter avoiding counter-trend trades

name = "12h_WilliamsAlligator_1dEMA34_Volume_v1"
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
    # Median price = (H + L) / 2
    median_price = (high + low) / 2.0
    
    # JAWS (Blue): 13-period SMMA, shifted 8 bars forward
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaws = np.roll(jaws, 8)
    jaws[:8] = np.nan
    
    # TEETH (Red): 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # LIPS (Green): 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
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
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: LIPS crosses above TEETH + price > 1d EMA34 + volume confirm
            if lips[i] > teeth[i] and lips[i-1] <= teeth[i-1] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: LIPS crosses below TEETH + price < 1d EMA34 + volume confirm
            elif lips[i] < teeth[i] and lips[i-1] >= teeth[i-1] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: LIPS crosses below JAWS (trend weakening)
            if lips[i] < jaws[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: LIPS crosses above JAWS (trend weakening)
            if lips[i] > jaws[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals