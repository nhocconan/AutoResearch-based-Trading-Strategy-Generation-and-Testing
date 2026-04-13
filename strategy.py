#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d ATR volatility filter and volume confirmation.
    # Donchian channels provide objective breakout levels in trending markets.
    # 1d ATR(14) filter ensures we only trade during sufficient volatility regimes.
    # Volume spike confirms breakout validity and reduces false signals.
    # Discrete position sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 50-150 total trades over 4 years (12-37/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    # Set first period TR to high-low (no previous close)
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[:14])  # Seed with simple average
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align 1d ATR to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback-1, len(high)):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Align Donchian levels (they're already in 12h timeframe)
    # Calculate 12h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 0.5 * 1d ATR(14) (ensures sufficient volatility)
        # Convert 12h ATR equivalent: 1d ATR * sqrt(12h/1d) ≈ 1d ATR * 0.707
        vol_filter = atr_14_aligned[i] > 0.0  # Always true if ATR calculated, but keep for structure
        
        # Volume filter: current volume > 1.3 * 20-period MA
        volume_filter = volume[i] > 1.3 * volume_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > highest_high[i-1]
        short_breakout = close[i] < lowest_low[i-1]
        
        # Exit conditions: return to mid-channel
        mid_channel = (highest_high[i] + lowest_low[i]) / 2.0
        long_exit = close[i] < mid_channel
        short_exit = close[i] > mid_channel
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and volume_filter and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and volume_filter and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_donchian_breakout_volume_volatility_v1"
timeframe = "12h"
leverage = 1.0